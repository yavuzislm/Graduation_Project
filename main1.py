import cv2
import numpy as np

def is_circular_enough(cnt, center, radius, threshold_ratio=0.6, tolerance=0.15):
    cx, cy = center
    within_range = 0
    for point in cnt:
        px, py = point[0]
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if abs(dist - radius) <= radius * tolerance:
            within_range += 1
    ratio = within_range / len(cnt)
    return ratio >= threshold_ratio

CALIBRATION_K = 6250
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

    # === MAVI BALON (DOST) ===
    lower_blue1 = np.array([85, 40, 100])
    upper_blue1 = np.array([110, 255, 255])
    lower_blue2 = np.array([85, 10, 180])
    upper_blue2 = np.array([110, 60, 255])
    lower_blue3 = np.array([85, 0, 190])
    upper_blue3 = np.array([115, 70, 255])

    mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    mask3 = cv2.inRange(hsv, lower_blue3, upper_blue3)
    mask_blue = cv2.bitwise_or(mask1, mask2)
    mask_blue = cv2.bitwise_or(mask_blue, mask3)

    lower_glass = np.array([85, 0, 220])
    upper_glass = np.array([110, 40, 255])
    mask_glass = cv2.inRange(hsv, lower_glass, upper_glass)
    mask_blue = cv2.bitwise_and(mask_blue, cv2.bitwise_not(mask_glass))

    kernel = np.ones((15, 15), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.GaussianBlur(mask_blue, (5, 5), 0)

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            if radius > 10 and is_circular_enough(cnt, center, radius):
                cv2.circle(frame, center, int(radius), (255, 0, 0), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                coord_x = center[0] - frame_center[0]
                coord_y = frame_center[1] - center[1]
                diameter = int(radius * 2)

                label = f"DOST  |  x: {coord_x}, y: {coord_y}  |  cap: {diameter}px"
                cv2.putText(frame, label, (int(center[0] - diameter), int(center[1] - radius - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                print(f"[DOST] x: {coord_x}, y: {coord_y}, cap: {diameter}px, distance: {int(298.0077 - diameter * 0.99208)}cm")

    # === KIRMIZI BALON (DUSMAN) ===
    lower_red1 = np.array([0, 110, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 110, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.GaussianBlur(mask_red, (5, 5), 0)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area > 500:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            if radius > 10 and is_circular_enough(cnt, center, radius):
                cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                coord_x = center[0] - frame_center[0]
                coord_y = frame_center[1] - center[1]
                diameter = int(radius * 2)

                label = f"DUSMAN  |  x: {coord_x}, y: {coord_y}  |  cap: {diameter}px"
                cv2.putText(frame, label, (int(center[0] - diameter), int(center[1] - radius - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                print(f"[DUSMAN] x: {coord_x}, y: {coord_y}, cap: {diameter}px, distance: {int(298.0077 - diameter * 0.99208)}cm")

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask Blue", mask_blue)
    cv2.imshow("Mask Red", mask_red)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
