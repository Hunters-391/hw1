# 嵌入式影像處理作業一:人臉瞳孔偵測
載入人臉模型
偵測臉 → 偵測眼睛
在眼睛內找「黑色區域」＝瞳孔
import cv2
import numpy as np

# 讀取圖片
img = cv2.imread("face.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 載入模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 偵測臉
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

    for (ex, ey, ew, eh) in eyes:
        eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_color = roi_color[ey:ey+eh, ex:ex+ew]

        # 👉 只取眼睛中間（避免眉毛）
        eye_gray = eye_gray[int(eh*0.25):int(eh*0.75), :]
        eye_color = eye_color[int(eh*0.25):int(eh*0.75), :]

        # 模糊
        blur = cv2.GaussianBlur(eye_gray, (9, 9), 0)

        # Hough Circle 偵測瞳孔
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=15,
            minRadius=3,
            maxRadius=20
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for (cx, cy, r) in circles[0, :]:
                cv2.circle(eye_color, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(eye_color, (cx, cy), 2, (0, 0, 255), 3)

# 顯示結果
cv2.imshow("Pupil Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
           
cap.release()
cv2.destroyAllWindows()

![顯示結果](./face.png)
