import cv2
import time
import os
import numpy as np

# ==========================================
# 1) CAMERA INITIALIZATION
# ==========================================
# Using your exact 1640x1232 GStreamer pipeline
gst_pipeline = (
    "libcamerasrc ! "
    "video/x-raw, width=820, height=616, format=YUY2, framerate=40/1 ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=True max-buffers=1 sync=False"
)

print(f"Opening Camera Pipeline...\n{gst_pipeline}")
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Failed to open camera.")

# ==========================================
# 2) DIRECTORY SETUP (Absolute Path Fix)
# ==========================================
# Dynamically grab the exact folder this python script lives in
current_folder = os.path.dirname(os.path.abspath(__file__))

# Build the save directory right next to the script
save_dir = os.path.join(current_folder, "calibration_images_219")
os.makedirs(save_dir, exist_ok=True)

print(f"Images will be saved to: {save_dir}")
print("Controls: Press 's' to save an image, 'q' to quit.")

# ==========================================
# 3) MAIN CAPTURE LOOP
# ==========================================
count = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to read from camera.")
            break

        # --- DISPLAY ---
        # Scale down the 1640x1232 image by 50% so it fits on your screen comfortably
        display = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # Add a simple HUD to show how many images you've captured
        cv2.putText(display, f"Captured: {count} | Press 's' to save", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Calibration Capture", display)

        # --- CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the ORIGINAL, unscaled frame for maximum calibration accuracy
            timestamp = int(time.time())
            filename = os.path.join(save_dir, f"calib_frame_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"[{count}] Saved: {filename}")
            
            # Flash the screen briefly to confirm capture
            flash = np.ones_like(display) * 255
            cv2.imshow("Calibration Capture", flash)
            cv2.waitKey(50)

except KeyboardInterrupt:
    print("\nStopping capture...")
finally:
    cap.release()
    cv2.destroyAllWindows()