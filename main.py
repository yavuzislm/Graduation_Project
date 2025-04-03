import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # === MAVİ BALON (DOST) ===
    lower_light_blue = np.array([80, 20, 100])
    upper_light_blue = np.array([110, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)

    # Mavi maske temizliği
    kernel = np.ones((7, 7), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.GaussianBlur(mask_blue, (5, 5), 0)

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area_blue = 0
    best_cnt_blue = None

    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > max_area_blue and len(cnt) >= 5:
            max_area_blue = area
            best_cnt_blue = cnt

    if best_cnt_blue is not None:
        ellipse = cv2.fitEllipse(best_cnt_blue)
        (x, y), (MA, ma), angle = ellipse
        aspect_ratio = max(MA, ma) / (min(MA, ma) + 1e-5)
        if 0.3 < aspect_ratio < 3.0:
            cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(frame, "DOST", (int(x - 30), int(y - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # === KIRMIZI BALON (DÜŞMAN) ===
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Kırmızı maske temizliği
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.GaussianBlur(mask_red, (5, 5), 0)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area_red = 0
    best_cnt_red = None

    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 300:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
            if circularity > 0.7 and area > max_area_red:
                max_area_red = area
                best_cnt_red = cnt

    if best_cnt_red is not None:
        (x, y), radius = cv2.minEnclosingCircle(best_cnt_red)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 0, 255), 2)
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        cv2.putText(frame, "DUSMAN", (int(x - radius), int(y - radius - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"Kırmızı balon (DÜŞMAN) merkezi: x={int(x)}, y={int(y)}")

    # Görüntüleme
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask Blue", mask_blue)
    cv2.imshow("Mask Red", mask_red)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
