import cv2
import numpy as np

pointSize = 5
pointColor = [(0,255,0),(255,0,0),(0,0,255)]
sampleStep = 10

windowName = "img"  
windowSize = [800, 600]  
imgOrigin = cv2.imread("test/images/2.jpg") 

pointsList = []
choosed = False
choosedPointIndex = [-1,-1]

def contourExtract(mask):
    global pointsList
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    tmp_layer = []
    for i in range(len(contours)):
        j = 0
        cnt = contours[i]
        if len(contours[i]) > 100: # TODO:
            while 1:
                tmp_layer.append((cnt[j][0][0],cnt[j][0][1]))
                j += sampleStep
                if j >= len(cnt):
                    break
            pointsList.append(tmp_layer)

def detectChoose(x,y):
    global choosedPointIndex
    for i in range(len(pointsList)):
        cnt = pointsList[i]
        for j in range(len(cnt)):
            point = cnt[j]
            if (x > point[0] - pointSize and x < point[0] + pointSize) and (y > point[1] - pointSize and y < point[1] + pointSize):
                choosedPointIndex = [i,j]
                return True
    return False

def mouse(event, x, y, flags, param):
    global pointsList, choosedPointIndex, choosed

    if event == cv2.EVENT_LBUTTONDOWN:
        choosed = detectChoose(x,y)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        if choosed:
            [i,j] = choosedPointIndex
            pointsList[i][j] = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        if choosed:
            choosedPointIndex = [-1,-1]
            choosed = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        choosed = detectChoose(x,y)
        if choosed:
            [i,j] = choosedPointIndex
            del pointsList[i][j]
        choosed = False
        choosedPointIndex = [-1,-1]
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        choosed = detectChoose(x,y)
        if choosed:
            [i,j] = choosedPointIndex
            P1 = pointsList[i][j]
            if j == 0:
                P2 = pointsList[i][len(pointsList[i])-1]
            else:
                P2 = pointsList[i][j-1]
            pointsList[i].insert(j,((P1[0] + P2[0])//2, (P1[1] + P2[1])//2))
        choosed = False
        choosedPointIndex = [-1,-1]

    curImg = imgOrigin.copy()
    draw(curImg, pointsList) 
    

# mode 1 for polys 2 for splines
def draw(img, points, mode=1):
    for i in range(len(points)):
        cnt = points[i]
        drawColor = pointColor[i%3]
        oppoColor = (255 - drawColor[0],255 - drawColor[1],255 - drawColor[2])
        if mode == 1:
            # draw the points
            for j in range(len(cnt)):
                if [i,j] == choosedPointIndex and choosedPointIndex != [-1,-1]:
                    cv2.circle(img, cnt[j], 3, oppoColor)
                else:
                    cv2.circle(img, cnt[j], 3, drawColor)
            # draw the lines
            for j in range(len(cnt)):
                if j == len(cnt)-1:
                    cv2.line(img, cnt[j], cnt[0], oppoColor)
                else:
                    cv2.line(img, cnt[j], cnt[j+1], oppoColor)
        elif mode == 2:
            #TODO
    cv2.imshow(windowName, img)

# GraphCut on current Pointlist
def refine(mode=1):
    if mode == 1:

    elif mode == 2:
        #TODO



if __name__ == "__main__":
    mask = cv2.imread("test/masks/2.png", cv2.IMREAD_GRAYSCALE)
    contourExtract(mask)
    cv2.namedWindow(windowName, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(windowName, windowSize[0], windowSize[1])
    cv2.imshow(windowName, imgOrigin)
    cv2.setMouseCallback(windowName, mouse)
    cv2.waitKey()
    cv2.destroyAllWindows()
    