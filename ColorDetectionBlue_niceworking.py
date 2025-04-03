import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Genişletilmiş açık mavi / soluk turkuaz renk aralığı
    lower_light_blue = np.array([80, 20, 100])
    upper_light_blue = np.array([110, 255, 255])

    mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    # Maske temizliği
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # En büyük konturu bul
    max_area = 0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and len(cnt) >= 5:
            max_area = area
            best_cnt = cnt

    if best_cnt is not None:
        ellipse = cv2.fitEllipse(best_cnt)
        (x, y), (MA, ma), angle = ellipse
        aspect_ratio = max(MA, ma) / (min(MA, ma) + 1e-5)

        if 0.3 < aspect_ratio < 3.0:
            cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(frame, "BALON", (int(x - 30), int(y - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            print(f"Balon merkezi: x={int(x)}, y={int(y)}")

    # Görüntüleme
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
