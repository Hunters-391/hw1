# 嵌入式影像處理作業一:人臉瞳孔偵測
載入人臉模型
偵測臉 → 偵測眼睛
在眼睛內找「黑色區域」＝瞳孔
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 讀圖（本地或Colab上傳都可）
img = cv2.imread('face.jpg')  # 改成你的圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 載入模型（OpenCV內建）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 偵測人臉
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

pupil_centers = []

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # 偵測眼睛
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_color = roi_color[ey:ey+eh, ex:ex+ew]

        # 降噪 + 二值化
        blur = cv2.GaussianBlur(eye_gray, (7,7), 0)
        _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY_INV)

        # 找輪廓（瞳孔）
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 50:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                # ⭐ 轉回整張圖座標
                center = (int(cx + ex + x), int(cy + ey + y))
                radius = int(radius)

                pupil_centers.append(center)

                cv2.circle(img, center, radius, (0,255,0), 2)
                break

# ⭐ 瞳孔距離計算
if len(pupil_centers) >= 2:
    # 依 x 排序（確保左右眼正確）
    pupil_centers = sorted(pupil_centers, key=lambda p: p[0])

    left_eye = pupil_centers[0]
    right_eye = pupil_centers[1]

    # 計算距離
    distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    # 畫線
    cv2.line(img, left_eye, right_eye, (0,255,255), 2)

    # 顯示距離
    cv2.putText(img, f"PD: {int(distance)} px",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2)

# 顯示結果
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
![顯示結果](./face.png)
