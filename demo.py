import cv2
import numpy as np

# The mouse location
ix,iy = 0,0

class Point(object):
    def __init__(self,x,y):
        self.xy = [x,y]     


back_img = cv2.imread("test/images/1.jpg")
fore_img = cv2.imread("test/masks/1.png", cv2.IMREAD_GRAYSCALE)
size = fore_img.shape
contours, hierarchy = cv2.findContours(fore_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#temp = np.ones(size,np.uint8)*255
#cv2.drawContours(temp,contours,-1,(0,255,0),2)

#cv2.imshow("contours",temp)
for i in range(len(contours)):
    cnt = contours[i]
    epsilon = 0.00001 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # cv2.drawContours(fore_img, approx, -1, (200, 200, 255), 3)
    # cv2.polylines(fore_img, [approx], True, (150, 0, 150), 3)
    for j in range(len(cnt)):
        cv2.circle(back_img, (cnt[j][0][0],cnt[j][0][1]), 3, (0,255,0))
    
cv2.imshow("polys",back_img)



cv2.waitKey(0)
cv2.destroyAllWindows()