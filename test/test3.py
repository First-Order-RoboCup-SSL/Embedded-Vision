import cv2
import time
import numpy as np
from collections import deque
from numba import njit

# ==========================================
# 1) NUMBA ACCELERATED CIRCLE FIT
# ==========================================

@njit(fastmath=True, cache=True)
def get_circle_model(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    D = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6:
        return np.full(3, np.nan)

    ux = ((x1 * x1 + y1 * y1) * (y2 - y3) +
          (x2 * x2 + y2 * y2) * (y3 - y1) +
          (x3 * x3 + y3 * y3) * (y1 - y2)) / D

    uy = ((x1 * x1 + y1 * y1) * (x3 - x2) +
          (x2 * x2 + y2 * y2) * (x1 - x3) +
          (x3 * x3 + y3 * y3) * (x2 - x1)) / D

    r = np.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
    return np.array([ux, uy, r])

@njit(fastmath=True, cache=True)
def fit_circle_ransac_numba(points, last_cx, last_cy,
                            center_thresh=50.0, n_iters=100,
                            inlier_thresh=2.0, min_inliers=40):
    N = points.shape[0]
    if N < min_inliers:
        return np.full(3, np.nan)

    best_count = -1
    best_model = np.full(3, np.nan)
    center_thresh_sq = center_thresh * center_thresh

    for _ in range(n_iters):
        idx1 = np.random.randint(0, N)
        idx2 = np.random.randint(0, N)
        idx3 = np.random.randint(0, N)
        if idx1 == idx2 or idx1 == idx3 or idx2 == idx3:
            continue

        model = get_circle_model(points[idx1], points[idx2], points[idx3])
        if np.isnan(model[0]):
            continue

        cx, cy, r = model
        if r < 5.0 or r > 200.0:
            continue

        if not np.isnan(last_cx):
            dist_sq = (cx - last_cx) ** 2 + (cy - last_cy) ** 2
            if dist_sq > center_thresh_sq:
                continue

        count = 0
        for i in range(N):
            dx = points[i, 0] - cx
            dy = points[i, 1] - cy
            d = np.sqrt(dx * dx + dy * dy)
            if abs(d - r) < inlier_thresh:
                count += 1

        if count > best_count:
            best_count = count
            best_model = model

    if best_count < min_inliers:
        return np.full(3, np.nan)
    return best_model

# ==========================================
# 2) SAFE ZONE LOGIC
# ==========================================

def is_in_safe_zone(cx, cy, width, height, margin_top, margin_bottom, corner_radius):
    if cy < margin_top:
        return False
    if cy > (height - margin_bottom):
        return False

    t = cy / float(height)
    current_side_margin = margin_top + (margin_bottom - margin_top) * t

    if cx < current_side_margin:
        return False
    if cx > (width - current_side_margin):
        return False

    dist_sq_bl = cx ** 2 + (cy - height) ** 2
    if dist_sq_bl < corner_radius ** 2:
        return False

    dist_sq_br = (cx - width) ** 2 + (cy - height) ** 2
    if dist_sq_br < corner_radius ** 2:
        return False

    return True

# ==========================================
# 3) YOUR WORKING OPENCV INIT + FPS LOOP
# ==========================================

W, H = 640, 480

gst = (
    "libcamerasrc ! "
    "video/x-raw,format=BGR,width=640,height=480,framerate=60/1 ! "
    "videoconvert ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError("Failed to open camera via GStreamer/libcamerasrc")

print("Backend:", cap.getBackendName())

# Warm up numba
fit_circle_ransac_numba(np.array([[10, 10], [20, 10], [15, 20]], dtype=np.float64), np.nan, np.nan)

# FPS tracking
t_prev = time.perf_counter()
dt_hist = deque(maxlen=120)  # smoothing window

# Tracker params
MIN_AREA_PIXELS = 100
MARGIN_TOP = 4
MARGIN_BOTTOM = 40
CORNER_RADIUS = 140
prev_cx, prev_cy = np.nan, np.nan

# Red HSV ranges (OpenCV Hue: 0..180)
lower_red1 = np.array([0,   180, 70], dtype=np.uint8)
upper_red1 = np.array([10,  255, 255], dtype=np.uint8)
lower_red2 = np.array([170, 180, 70], dtype=np.uint8)
upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("read failed")
        break

    # --- FPS calc (your logic) ---
    t_now = time.perf_counter()
    dt = t_now - t_prev
    t_prev = t_now

    if dt > 0:
        dt_hist.append(dt)
        inst_fps = 1.0 / dt
        avg_dt = sum(dt_hist) / len(dt_hist)
        smooth_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
    else:
        inst_fps = 0.0
        smooth_fps = 0.0

    # Use actual frame size (in case pipeline changes)
    Hf, Wf = frame.shape[:2]

    # ==========================================
    # 4) RGB -> HSV + RED MASK (as requested)
    # ==========================================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphology denoising
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components to get largest blob
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    final_circle = None
    mode = "Scanning"

    if num_labels > 1:
        largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_area = stats[largest_label_idx, cv2.CC_STAT_AREA]

        if largest_area > MIN_AREA_PIXELS:
            blob_mask = (labels == largest_label_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (mx, my), mr = cv2.minEnclosingCircle(largest_contour)

                is_safe = is_in_safe_zone(mx, my, Wf, Hf, MARGIN_TOP, MARGIN_BOTTOM, CORNER_RADIUS)

                if is_safe:
                    final_circle = (mx, my, mr)
                    mode = "Stable (MEC)"
                    prev_cx, prev_cy = mx, my
                else:
                    pts_xy = largest_contour.reshape(-1, 2).astype(np.float64)
                    res = fit_circle_ransac_numba(
                        pts_xy, prev_cx, prev_cy,
                        center_thresh=50.0, n_iters=100,
                        inlier_thresh=2.0, min_inliers=40
                    )
                    if not np.isnan(res[0]):
                        final_circle = (res[0], res[1], res[2])
                        mode = "Edge (RANSAC)"
                        prev_cx, prev_cy = res[0], res[1]
                    else:
                        mode = "RANSAC Failed"

    # ==========================================
    # 5) DISPLAY (convert RGB -> BGR for imshow)
    # ==========================================
    display = frame.copy()

    # Safe zone trapezoid lines
    cv2.line(display, (MARGIN_TOP, MARGIN_TOP), (Wf - MARGIN_TOP, MARGIN_TOP), (0, 255, 255), 2)
    cv2.line(display, (MARGIN_BOTTOM, Hf - MARGIN_BOTTOM), (Wf - MARGIN_BOTTOM, Hf - MARGIN_BOTTOM), (0, 255, 255), 2)
    cv2.line(display, (MARGIN_TOP, MARGIN_TOP), (MARGIN_BOTTOM, Hf - MARGIN_BOTTOM), (0, 255, 255), 2)
    cv2.line(display, (Wf - MARGIN_TOP, MARGIN_TOP), (Wf - MARGIN_BOTTOM, Hf - MARGIN_BOTTOM), (0, 255, 255), 2)

    # Corner exclusion circles
    cv2.circle(display, (0, Hf), CORNER_RADIUS, (0, 0, 255), 2)
    cv2.circle(display, (Wf, Hf), CORNER_RADIUS, (0, 0, 255), 2)

    if final_circle is not None:
        cx, cy, r = final_circle
        color = (0, 255, 0) if "MEC" in mode else (0, 165, 255)
        cv2.circle(display, (int(cx), int(cy)), int(r), color, 3)
        cv2.circle(display, (int(cx), int(cy)), 5, color, -1)

    # your FPS overlay (kept)
    cv2.putText(display, f"FPS inst: {inst_fps:6.1f}  avg: {smooth_fps:6.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, f"Mode: {mode}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("cam", display)

    # print once per second (your style)
    if len(dt_hist) > 0 and (t_now % 1.0) < dt:
        print(f"FPS avg: {smooth_fps:.2f}  (inst {inst_fps:.2f}) | Mode: {mode}")

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
