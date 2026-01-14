import time
import cv2
import numpy as np
import threading
from picamera2 import Picamera2
from numba import njit


@njit(fastmath=True, cache=True)
def get_circle_model(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6: return np.full(3, np.nan)
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / D
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / D
    r = np.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
    return np.array([ux, uy, r])


@njit(fastmath=True, cache=True)
def fit_circle_ransac_numba(points, last_cx, last_cy, center_thresh=50.0, n_iters=80, inlier_thresh=2.0,
                            min_inliers=50):
    N = points.shape[0]
    if N < min_inliers: return np.full(3, np.nan)

    best_count = -1
    best_model = np.full(3, np.nan)
    center_thresh_sq = center_thresh ** 2

    for _ in range(n_iters):
        idx1, idx2, idx3 = np.random.randint(0, N), np.random.randint(0, N), np.random.randint(0, N)
        if idx1 == idx2 or idx1 == idx3 or idx2 == idx3: continue

        model = get_circle_model(points[idx1], points[idx2], points[idx3])
        if np.isnan(model[0]): continue
        cx, cy, r = model

        if r < 5 or r > 200: continue

        if not np.isnan(last_cx):
            dist_sq = (cx - last_cx) ** 2 + (cy - last_cy) ** 2
            if dist_sq > center_thresh_sq: continue

        dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        count = 0
        for i in range(N):
            if abs(dists[i] - r) < inlier_thresh: count += 1

        if count > best_count:
            best_count = count
            best_model = model

    if best_count < min_inliers: return np.full(3, np.nan)
    return best_model


def is_in_safe_zone(cx, cy, width, height, margin_top, margin_bottom, corner_radius):
    """
    Checks if center (cx, cy) is safe.
    1. Must be within the Trapezoid defined by margin_top/bottom.
    2. Must be OUTSIDE the corner exclusion circles at Top-Left and Top-Right.
    """

    # 1. Vertical Bounds (Hard Limits)
    if cy < margin_top: return False
    if cy > (height - margin_bottom): return False

    # 2. Dynamic Side Margin (Trapezoid Logic)
    t = cy / height
    current_side_margin = margin_top + (margin_bottom - margin_top) * t

    if cx < current_side_margin: return False
    if cx > (width - current_side_margin): return False

    # 3. Corner Radius Exclusion
    # We define a "Danger Zone" circle centered exactly at (0,0) and (Width,0).
    # If distance to corner < corner_radius, it is UNSAFE (trigger RANSAC).

    # Top-Left Corner (0,0)
    dist_sq_tl = cx ** 2 + cy ** 2
    if dist_sq_tl < corner_radius ** 2:
        return False  # Too close to top-left corner

    # Top-Right Corner (Width, 0)
    dist_sq_tr = (cx - width) ** 2 + cy ** 2
    if dist_sq_tr < corner_radius ** 2:
        return False  # Too close to top-right corner

    return True



class CameraThread:
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "YUV420"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.picam2.set_controls({
            "AeEnable": False, "ExposureTime": 8000,
            "AnalogueGain": 10.0, "FrameDurationLimits": (8333, 20000)
        })
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            frame = self.picam2.capture_array()
            with self.lock: self.latest_frame = frame

    def get_frame(self):
        with self.lock: return self.latest_frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.picam2.stop()



def main():
    HEADLESS = False
    print("Initializing...")
    cam = CameraThread()
    time.sleep(1.0)

    # Trigger Numba compilation
    fit_circle_ransac_numba(np.array([[10, 10], [20, 10], [15, 20]], dtype=np.float64), np.nan, np.nan)
    print("System Ready.")

    RED_THRESHOLD = 160
    MIN_AREA_PIXELS = 100

    FRAME_W = 320
    FRAME_H = 240

    # --- CONFIG ---
    MARGIN_TOP = 20
    MARGIN_BOTTOM = 2
    CORNER_RADIUS = 70

    prev_cx, prev_cy = np.nan, np.nan
    fps = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            raw_yuv = cam.get_frame()
            if raw_yuv is None: continue

            v_channel = raw_yuv[600:720, :].reshape((240, 320))
            mask = (v_channel > RED_THRESHOLD).astype(np.uint8) * 255

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            final_circle = None
            mode = "Scanning"

            if num_labels > 1:
                largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                largest_area = stats[largest_label_idx, cv2.CC_STAT_AREA]

                if largest_area > MIN_AREA_PIXELS:
                    blob_mask = (labels == largest_label_idx).astype(np.uint8)
                    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        largest_contour = contours[0]
                        (mx, my), mr = cv2.minEnclosingCircle(largest_contour)

                        # --- SAFE ZONE CHECK ---
                        is_safe = is_in_safe_zone(mx, my, FRAME_W, FRAME_H, MARGIN_TOP, MARGIN_BOTTOM, CORNER_RADIUS)

                        if is_safe:
                            final_circle = (mx, my, mr)
                            mode = "Stable (MEC)"
                            prev_cx, prev_cy = mx, my
                        else:
                            pts_xy = largest_contour.reshape(-1, 2).astype(np.float64)
                            res = fit_circle_ransac_numba(
                                pts_xy, prev_cx, prev_cy,
                                center_thresh=50.0, n_iters=100, min_inliers=40
                            )

                            if not np.isnan(res[0]):
                                final_circle = res
                                mode = "Edge (RANSAC)"
                                prev_cx, prev_cy = res[0], res[1]
                            else:
                                mode = "RANSAC Failed"

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            if HEADLESS:
                if final_circle is not None:
                    print(f"[{mode}] FPS: {fps:.1f} | ({final_circle[0]:.1f}, {final_circle[1]:.1f})")
            else:
                display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # --- VISUALIZE SAFE ZONE ---
                # Draw Trapezoid Lines
                cv2.line(display_img, (MARGIN_TOP, MARGIN_TOP), (FRAME_W - MARGIN_TOP, MARGIN_TOP), (50, 50, 50), 1)
                cv2.line(display_img, (MARGIN_BOTTOM, FRAME_H - MARGIN_BOTTOM),
                         (FRAME_W - MARGIN_BOTTOM, FRAME_H - MARGIN_BOTTOM), (50, 50, 50), 1)
                cv2.line(display_img, (MARGIN_TOP, MARGIN_TOP), (MARGIN_BOTTOM, FRAME_H - MARGIN_BOTTOM), (50, 50, 50),
                         1)
                cv2.line(display_img, (FRAME_W - MARGIN_TOP, MARGIN_TOP),
                         (FRAME_W - MARGIN_BOTTOM, FRAME_H - MARGIN_BOTTOM), (50, 50, 50), 1)

                # Draw Corner Exclusion Circles (Red Arcs)
                cv2.circle(display_img, (0, 0), CORNER_RADIUS, (0, 0, 100), 1)
                cv2.circle(display_img, (FRAME_W, 0), CORNER_RADIUS, (0, 0, 100), 1)

                if final_circle is not None:
                    cx, cy, r = final_circle
                    color = (0, 255, 0) if "MEC" in mode else (0, 165, 255)
                    cv2.circle(display_img, (int(cx), int(cy)), int(r), color, 2)
                    cv2.circle(display_img, (int(cx), int(cy)), 3, color, -1)

                cv2.putText(display_img, f"FPS: {fps:.1f} | {mode}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Trapezoid + Corner Tracker", display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.stop()
        if not HEADLESS: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()