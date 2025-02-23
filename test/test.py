import cv2

cap = cv2.VideoCapture(0)  # 0 对应 /dev/video0

if not cap.isOpened():
    print("Fail to open the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive the frame")
        break

    frame = cv2.flip(frame, 0)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
        break

cap.release()
cv2.destroyAllWindows()




