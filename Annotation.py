import cv2
import numpy as np
import GraphCut as GC

class App:

    pointSize = 5
    pointColor = [(0,255,0),(255,0,0),(0,0,255)]
    sampleStep = 10

    windowName = "img"  
    windowSize = [800, 600]  

    pointsList = []
    choosed = False
    choosedPointIndex = [-1,-1]

    def __init__(self, input, mask, output):
        self.input = input
        self.mask = mask
        self.output = output
        self.imgOrigin = cv2.imread(self.input) 
        self.GC = GC.GCGraph(self.imgOrigin)

    def contourExtract(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            tmp_layer = []
            j = 0
            cnt = contours[i]
            if len(contours[i]) > 50: # TODO: bigger sample rate for arc curve
                while 1:
                    tmp_layer.append(cnt[j])
                    j += self.sampleStep
                    if j >= len(cnt):
                        break
                tmp_layer = np.array(tmp_layer)
                self.pointsList.append(tmp_layer)

    def detectChoose(self, x, y):
        for i in range(len(self.pointsList)):
            cnt = self.pointsList[i]
            for j in range(len(cnt)):
                point = cnt[j][0]
                if (x > point[0] - self.pointSize and x < point[0] + self.pointSize) and (y > point[1] - self.pointSize and y < point[1] + self.pointSize):
                    self.choosedPointIndex = [i,j]
                    return True
        return False

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.choosed = self.detectChoose(x,y)
        # Left button down and drag
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.choosed:
                [i,j] = self.choosedPointIndex
                self.pointsList[i][j][0][0] = x
                self.pointsList[i][j][0][1] = y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.choosed:
                self.choosedPointIndex = [-1,-1]
                self.choosed = False
        # Right button to delete
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.choosed = self.detectChoose(x,y)
            if self.choosed:
                [i,j] = self.choosedPointIndex
                tmp = self.pointsList[i].tolist()
                del tmp[j]
                del self.pointsList[i]
                self.pointsList.insert(i,np.array(tmp))
            self.choosed = False
            self.choosedPointIndex = [-1,-1]
        # Left button double click to add
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.choosed = self.detectChoose(x,y)
            if self.choosed:
                [i,j] = self.choosedPointIndex
                P1 = self.pointsList[i][j]
                if j == 0:
                    P2 = self.pointsList[i][len(self.pointsList[i])-1]
                else:
                    P2 = self.pointsList[i][j-1]
                v1 = (P1[0][0] + P2[0][0])//2
                v2 = (P1[0][1] + P2[0][1])//2
                tmp = self.pointsList[i].tolist()
                tmp.insert(j,P1.copy())
                tmp[j][0][0] = v1
                tmp[j][0][1] = v2
                del self.pointsList[i]
                self.pointsList.insert(i,np.array(tmp))
            self.choosed = False
            self.choosedPointIndex = [-1,-1]
        # Middle Button click to refine contour
        elif event == cv2.EVENT_MBUTTONDOWN:
            img = self.genNewMask()
            cv2.imwrite(self.output,img)
        # Middle button double click to get mask
        elif event == cv2.EVENT_MBUTTONDBLCLK:
            self.refine(self.genNewMask())
            
        curImg = self.imgOrigin.copy()
        self.draw(curImg, self.pointsList) 
        
    # mode 1 for polys 2 for splines
    def draw(self, img, points, mode=1):
        for i in range(len(points)):
            cnt = points[i]
            drawColor = self.pointColor[i%3]
            oppoColor = (255 - drawColor[0],255 - drawColor[1],255 - drawColor[2])
            if mode == 1:
                # draw the points
                for j in range(len(cnt)):
                    if [i,j] == self.choosedPointIndex and self.choosedPointIndex != [-1,-1]:
                        cv2.circle(img, (cnt[j][0][0],cnt[j][0][1]), 3, oppoColor)
                    else:
                        cv2.circle(img, (cnt[j][0][0],cnt[j][0][1]), 3, drawColor)
                # draw the lines
                for j in range(len(cnt)):
                    if j == len(cnt)-1:
                        cv2.line(img, (cnt[j][0][0],cnt[j][0][1]), (cnt[0][0][0],cnt[0][0][1]), oppoColor)
                    else:
                        cv2.line(img, (cnt[j][0][0],cnt[j][0][1]), (cnt[j+1][0][0],cnt[j+1][0][1]), oppoColor)
            elif mode == 2:
                #TODO:
                pass
        cv2.imshow(self.windowName, img)

    def genNewMask(self):
        img = np.zeros(self.imgOrigin.shape)
        cv2.fillPoly(img,self.pointsList,(255,255,255))
        return img

    # GraphCut on current Pointslist
    def refine(self, mask, mode=1):
        if mode == 1:
            segments = self.GC.updateGraph(mask)
            [x,y,z] = self.imgOrigin.shape
            FGmask = np.ones((x,y), dtype=np.uint8) * 255
            for i,p in enumerate(segments):
                if p and (i<(x*y)):
                    row = i // x
                    col = i % x
                    FGmask[row][col] = 0
            cv2.imwrite('test.jpg',FGmask)
            print('Finish')
            # Recompute Contours        
            #self.pointsList = []
            #self.contourExtract(FGmask)
            
        elif mode == 2:
            #TODO:
            pass

    def run(self):
        mask = cv2.imread(self.mask, cv2.IMREAD_GRAYSCALE)
        self.contourExtract(mask)

        cv2.namedWindow(self.windowName, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.windowName, self.windowSize[0], self.windowSize[1])
        cv2.imshow(self.windowName, self.imgOrigin)
        cv2.setMouseCallback(self.windowName, self.mouse)
        cv2.waitKey()
        cv2.destroyAllWindows()
        