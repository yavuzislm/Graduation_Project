import cv2
import numpy as np
#Kısaltma yapmak için as np yi kullanıyoruz. Bu sayede numpy fonksiyonunu np dediğimizde başka yerde çağırabileceğiz.

cap = cv2.VideoCapture(0)

while True:
    flag,frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    cv2.imshow("bgr", frame)
    cv2.imshow("hsv", hsv)

    lower_green = np.array([35,100,100])
    upper_green = np.array([85,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow("mask", mask)

    result = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow("result", result)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()