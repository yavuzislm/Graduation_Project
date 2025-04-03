import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Daha spesifik kırmızı (balonun tonu)
    lower_red1 = np.array([0, 200, 200])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([170, 200, 200])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Gürültü temizleme
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                aspect_ratio = max(MA, ma) / (min(MA, ma) + 1e-5)
                if 0.5 < aspect_ratio < 2.0:
                    cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    cv2.putText(frame, "BALON", (int(x - 30), int(y - 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print(f"Balon merkezi: x={int(x)}, y={int(y)}")

    # Görüntüleme
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
