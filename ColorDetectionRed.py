import cv2
import numpy as np
#Kısaltma yapmak için as np yi kullanıyoruz. Bu sayede numpy fonksiyonunu np dediğimizde başka yerde çağırabileceğiz.

cap = cv2.VideoCapture(0)

while True:
    flag,frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    cv2.imshow("bgr", frame)
    cv2.imshow("hsv", hsv)
    
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,100,100])
    upper_red2 = np.array([180,255,255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    cv2.imshow("mask", mask)

    result = cv2.bitwise_and(frame, frame, mask = mask)
    cv2.imshow("result", result)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()