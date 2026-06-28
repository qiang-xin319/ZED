import cv2

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('myvideo.avi', fourcc, 20.0, (2560, 720))

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 手动调整大小，确保分辨率是 1280x640
    # frame = cv2.resize(frame, (1280, 480))

    out.write(frame)
    cv2.imshow('camera', frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
    if key == ord("w"):
        save_path = r"F:\ZED2i-ranging\Binocular-ranging-main\image\%d.png" % i
        cv2.imwrite(save_path, frame)  # 保存调整后的图像
        print(f"Save image {i} succeed at {save_path}")
        i += 1

cap.release()
out.release()
cv2.destroyAllWindows()
