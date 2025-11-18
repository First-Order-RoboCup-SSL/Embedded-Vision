import cv2
import numpy as np
import time

# ===================== Utilities =====================
def set_prop(cap, prop, value, name):
    ok = cap.set(prop, value)
    got = cap.get(prop)
    print(f"[SET] {name:<22} -> {value} (driver now reports {got}) {'✓' if ok else '✗'}")
    return ok, got

def try_get(cap, prop, name):
    v = cap.get(prop)
    if v == -1 or v is None:
        print(f"[GET] {name:<22} -> unsupported")
        return None
    print(f"[GET] {name:<22} -> {v}")
    return v

def auto_calibrate_and_lock(cap, warmup_seconds=2.0, report_every=0.5):
    print("\n=== Auto-calibration: enabling auto controls ===")
    # Exposure: V4L2 convention: 3=auto, 1=manual (OpenCV backends vary, but this works on many UVC cams)
    set_prop(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 3, "Auto Exposure (on)")
    set_prop(cap, cv2.CAP_PROP_AUTO_WB,      1, "Auto White Balance (on)")
    set_prop(cap, cv2.CAP_PROP_AUTOFOCUS,    1, "Auto Focus (on)")

    start = time.time()
    last_report = start
    frames = 0

    while time.time() - start < warmup_seconds:
        ret, _ = cap.read()
        if not ret:
            break
        frames += 1
        now = time.time()
        if now - last_report >= report_every:
            try_get(cap, cv2.CAP_PROP_EXPOSURE,       "Exposure")
            try_get(cap, cv2.CAP_PROP_GAIN,           "Gain")
            try_get(cap, cv2.CAP_PROP_WB_TEMPERATURE, "WB Temperature")
            try_get(cap, cv2.CAP_PROP_FOCUS,          "Focus")
            last_report = now

    print(f"Warm-up done, captured ~{frames} frames")

    settled = {
        "Exposure":       try_get(cap, cv2.CAP_PROP_EXPOSURE,       "Exposure"),
        "Gain":           try_get(cap, cv2.CAP_PROP_GAIN,           "Gain"),
        "WB Temperature": try_get(cap, cv2.CAP_PROP_WB_TEMPERATURE, "WB Temperature"),
        "Focus":          try_get(cap, cv2.CAP_PROP_FOCUS,          "Focus"),
        "Brightness":     try_get(cap, cv2.CAP_PROP_BRIGHTNESS,     "Brightness"),
        "Contrast":       try_get(cap, cv2.CAP_PROP_CONTRAST,       "Contrast"),
        "Saturation":     try_get(cap, cv2.CAP_PROP_SATURATION,     "Saturation"),
        "Sharpness":      try_get(cap, cv2.CAP_PROP_SHARPNESS,      "Sharpness"),
    }

    print("\n=== Locking controls to settled values ===")
    set_prop(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 1, "Auto Exposure (manual)")
    set_prop(cap, cv2.CAP_PROP_AUTO_WB,       0, "Auto White Balance (off)")
    set_prop(cap, cv2.CAP_PROP_AUTOFOCUS,     0, "Auto Focus (off)")

    for key, prop in [
        ("Exposure", cv2.CAP_PROP_EXPOSURE),
        ("Gain", cv2.CAP_PROP_GAIN),
        ("WB Temperature", cv2.CAP_PROP_WB_TEMPERATURE),
        ("Focus", cv2.CAP_PROP_FOCUS),
        ("Brightness", cv2.CAP_PROP_BRIGHTNESS),
        ("Contrast", cv2.CAP_PROP_CONTRAST),
        ("Saturation", cv2.CAP_PROP_SATURATION),
        ("Sharpness", cv2.CAP_PROP_SHARPNESS),
    ]:
        val = settled.get(key)
        if val is not None:
            set_prop(cap, prop, val, key)

    print("Lock complete.\n")

# ===================== Mouse HSV probe =====================
def mouse_callback(event, x, y, flags, param):
    """When you click on the window, print the HSV value at that pixel."""
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param.get("hsv_frame", None)
        if hsv_frame is not None and 0 <= y < hsv_frame.shape[0] and 0 <= x < hsv_frame.shape[1]:
            hsv_value = hsv_frame[y, x]
            print(f"HSV at (x={x}, y={y}): {hsv_value}")

# ===================== Red-ball detection =====================
def detect_red_ball(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # two red ranges in HSV
    lower_red1 = np.array([0, 135, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 135, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Morphological cleanup
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = {"center": None, "radius": None, "diff_x": None, "hsv": hsv}

    if contours:
        c = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            center = (int(cx), int(cy))
            radius_i = int(radius)
            cv2.circle(frame_bgr, center, radius_i, (0, 255, 0), 2)
            cv2.circle(frame_bgr, center, 4, (0, 0, 255), -1)
            diff_x = int(cx - frame_bgr.shape[1] / 2)
            info_text = f"Ball ({int(cx)}, {int(cy)}), r={radius_i}, dx={diff_x}"
            cv2.putText(frame_bgr, info_text, (center[0] - 60, max(20, center[1] - radius_i - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            result.update({"center": center, "radius": radius_i, "diff_x": diff_x})
    return frame_bgr, result

# ===================== ArUco detection =====================
def setup_aruco():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    try:
        parameters = cv2.aruco.DetectorParameters()
    except Exception:
        # Fallback for older OpenCV versions
        parameters = cv2.aruco.DetectorParameters_create()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    return detector

def detect_aruco_and_draw(frame_bgr, detector, cam_matrix, dist_coeffs, draw_axes=True, marker_length=0.187):
    corners, ids, rejected = detector.detectMarkers(frame_bgr)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame_bgr, corners, ids)
        if cam_matrix is not None and dist_coeffs is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, cam_matrix, dist_coeffs
            )
            if draw_axes:
                for i in range(len(ids)):
                    try:
                        cv2.drawFrameAxes(frame_bgr, cam_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length * 1.5)
                    except Exception:
                        # Some OpenCV builds expect (3,) not (1,3)
                        cv2.drawFrameAxes(frame_bgr, cam_matrix, dist_coeffs, rvecs[i][0], tvecs[i][0], marker_length * 1.5)
        else:
            # No intrinsics: still draw borders
            pass

    if rejected is not None and len(rejected) > 0:
        cv2.aruco.drawDetectedMarkers(frame_bgr, rejected, borderColor=(100, 0, 255))

    return frame_bgr, (corners, ids)

# ===================== Main =====================
def main():
    cam_id = 0
    # Nominal intrinsics (update with your real calibration if you have it)
    cam_matrix = np.array([[800, 0, 320],
                           [0, 800, 240],
                           [0,   0,   1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    marker_length = 0.187  # meters

    detector = setup_aruco()

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    # Choose codec/resolution/FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          60)

    # Initial auto-calibration + lock
    auto_calibrate_and_lock(cap, warmup_seconds=2.0)

    # Window + mouse HSV probe
    win_name = "Ball + ArUco (press a=auto-cal, f=flip, p=pose, q=quit)"
    cv2.namedWindow(win_name)
    param = {"hsv_frame": None}
    cv2.setMouseCallback(win_name, mouse_callback, param)

    # Toggles
    flip_vertical = False
    draw_pose_axes = True

    # FPS
    frame_count = 0
    fps = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed — exiting.")
            break

        if flip_vertical:
            frame = cv2.flip(frame, 0)

        # Red-ball detection
        frame, ball_info = detect_red_ball(frame)
        param["hsv_frame"] = ball_info.get("hsv")

        # ArUco detection
        tick = cv2.getTickCount()
        frame, _ = detect_aruco_and_draw(
            frame, detector, cam_matrix, dist_coeffs,
            draw_axes=draw_pose_axes, marker_length=marker_length
        )
        detect_ms = (cv2.getTickCount() - tick) / cv2.getTickFrequency() * 1000.0

        # HUD: resolution, FPS, detect time
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"ArUco detect: {detect_ms:.1f} ms", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # On-screen hint
        cv2.putText(frame, "Keys: a=auto-cal f=flip p=pose q/Esc=quit", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(win_name, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('a'):
            auto_calibrate_and_lock(cap, warmup_seconds=2.0)
        elif k == ord('f'):
            flip_vertical = not flip_vertical
            print(f"[INFO] Flip vertical: {flip_vertical}")
        elif k == ord('p'):
            draw_pose_axes = not draw_pose_axes
            print(f"[INFO] Draw pose axes: {draw_pose_axes}")
        elif k == ord('q') or k == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
