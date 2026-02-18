import cv2
import numpy as np

# ----------------------------
# 1. OPTIMIZED GSTREAMER PIPELINE
# ----------------------------
# - Source: 1640x1232 (Full FoV Binned Mode)
# - Format: YUY2 (As requested)
# - Scale:  Down to 640x480 for fast UI processing
# - Output: BGR (Compatible with OpenCV imshow/bitwise)
gst_pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=1640, height=1232, format=YUY2, framerate=40/1 ! "
    "videoscale ! "
    "video/x-raw, width=640, height=480 ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=True max-buffers=1 sync=False"
)

print(f"Opening Pipeline:\n{gst_pipeline}")
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera via GStreamer")

print("Backend:", cap.getBackendName())

# ----------------------------
# 2. UI SETUP (Unchanged)
# ----------------------------
def nothing(x): pass

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 520, 380)

# Trackbars
cv2.createTrackbar("H1 Low",  "Controls", 0,   179, nothing)
cv2.createTrackbar("H1 High", "Controls", 10,  179, nothing)
cv2.createTrackbar("H2 Low",  "Controls", 170, 179, nothing)
cv2.createTrackbar("H2 High", "Controls", 179, 179, nothing)

cv2.createTrackbar("S Low",   "Controls", 120, 255, nothing)
cv2.createTrackbar("S High",  "Controls", 255, 255, nothing)
cv2.createTrackbar("V Low",   "Controls", 70,  255, nothing)
cv2.createTrackbar("V High",  "Controls", 255, 255, nothing)

cv2.createTrackbar("Morph k", "Controls", 0,   15,  nothing)
cv2.createTrackbar("Erode",   "Controls", 0,   5,   nothing)
cv2.createTrackbar("Dilate",  "Controls", 0,   5,   nothing)

cv2.namedWindow("BGR | Mask | Masked", cv2.WINDOW_NORMAL)

# ----------------------------
# 3. MAIN LOOP
# ----------------------------
try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("read failed")
            break

        # Read slider values
        h1l = cv2.getTrackbarPos("H1 Low", "Controls")
        h1u = cv2.getTrackbarPos("H1 High", "Controls")
        h2l = cv2.getTrackbarPos("H2 Low", "Controls")
        h2u = cv2.getTrackbarPos("H2 High", "Controls")
        sl  = cv2.getTrackbarPos("S Low", "Controls")
        su  = cv2.getTrackbarPos("S High", "Controls")
        vl  = cv2.getTrackbarPos("V Low", "Controls")
        vu  = cv2.getTrackbarPos("V High", "Controls")
        mk  = cv2.getTrackbarPos("Morph k", "Controls")
        er  = cv2.getTrackbarPos("Erode", "Controls")
        di  = cv2.getTrackbarPos("Dilate", "Controls")

        # Clamp logic
        if h1l > h1u: h1l, h1u = h1u, h1l
        if h2l > h2u: h2l, h2u = h2u, h2l
        if sl  > su:  sl,  su  = su,  sl
        if vl  > vu:  vl,  vu  = vu,  vl

        # Conversion & Masking
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([h1l, sl, vl], dtype=np.uint8)
        upper1 = np.array([h1u, su, vu], dtype=np.uint8)
        lower2 = np.array([h2l, sl, vl], dtype=np.uint8)
        upper2 = np.array([h2u, su, vu], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphology
        if mk > 0:
            k = mk if mk % 2 == 1 else mk + 1
            kernel = np.ones((k, k), np.uint8)
            if er > 0:
                mask = cv2.erode(mask, kernel, iterations=er)
            if di > 0:
                mask = cv2.dilate(mask, kernel, iterations=di)

        masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Visualization
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = np.hstack([frame, mask_bgr, masked])

        txt = f"H1[{h1l},{h1u}] H2[{h2l},{h2u}] S[{sl},{su}] V[{vl},{vu}]"
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("BGR | Mask | Masked", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if key == ord('s'):
            print(f"Values: {txt}")

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
