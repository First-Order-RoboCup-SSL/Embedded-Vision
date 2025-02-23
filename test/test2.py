import cv2
import numpy as np
import time

hsv_frame = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param.get("hsv_frame", None)
        if hsv_frame is not None:
            # NB: Row -> y, Column -> x
            hsv_value = hsv_frame[y, x]
            print(f"HSV at (x={x}, y={y}): {hsv_value}")


def main():
    # we need cv2.VideoCapture("/dev/video0")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Where is the camera?")
        return
    
    cv2.namedWindow("Red Ball Detection")
    param = {"hsv_frame": None}
    cv2.setMouseCallback("Red Ball Detection", mouse_callback, param)
    
    frame_count = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Bye!")
            break

        frame = cv2.flip(frame, 0)
        # our robot need it

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        param["hsv_frame"] = hsv

        lower_red1 = np.array([0, 130, 170])    # [H=0, S=120, V=70]
        upper_red1 = np.array([10, 255, 255])  # [H=10, S=255, V=255]
        lower_red2 = np.array([160, 40, 140])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            center_x = x + w // 2
            center_y = y + h // 2

            frame_center_x = frame.shape[1] // 2
            diff_x = center_x - frame_center_x

            info_text = f"Center: ({center_x}, {center_y}), diff_x: {diff_x}"
            cv2.putText(frame, info_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution_text = f"Resolution: {width}x{height}"
        cv2.putText(frame, resolution_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
        cv2.imshow("Red Ball Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
