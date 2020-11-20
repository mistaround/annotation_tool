import numpy as np
import maxflow
from sklearn import mixture,svm
import math
import cv2

class GCGraph(object):
    def __init__(self,img):
        self.k = 5
        self.sigma = 1
        self.alpha = 2
        self.lam = 1000

        self.img = np.array(img)
        self.Im = self.img.reshape(-1,3)
        [x,y,z] = img.shape
        self.G = maxflow.Graph[int](x,y)
        self.nodes = self.G.add_nodes(x*y)
        self.sourceWeights = []
        self.sinkWeights = []

        self.init_lweights(x,y,self.k,self.sigma)

    def init_lweights(self,x,y,k,sigma):
        print ("Initializing weights")
        for i in range(0, x*y):
            # four neighboring pixels
            if i%x != 0: # for left pixels
                w = k*np.exp(-(np.linalg.norm(self.Im[i] - self.Im[i-1]))/self.sigma) # the cost function for two pixels is the frobenous norm between pixels
                self.G.add_edge(i, i-1, w, k-w)

            if (i+1)%x != 0: # for right pixels
                w = k*np.exp(-(np.linalg.norm(self.Im[i] - self.Im[i+1]))/self.sigma)
                self.G.add_edge(i, i+1, w, k-w)

            if i//x != 0: # for top pixels
                w = k*np.exp(-(np.linalg.norm(self.Im[i] - self.Im[i-x]))/self.sigma)
                self.G.add_edge(i, i-x, w, k-w)

            if i//x != y-1: # for bottom pixels
                w = k*np.exp(-(np.linalg.norm(self.Im[i] - self.Im[i+x]))/self.sigma)
                self.G.add_edge(i, i+x, w, k-w)
        print ("Finish")


    def is_surrounded(self,array,x,y,val,step = 1):
        flag = True
        shape = array.shape
        w,h = shape[0], shape[1]
        for i in range(1,step+1):
            if array[clamp(x,w)][clamp(y,h)][0] != val:
                flag = False
            if array[clamp(x-i,w)][clamp(y,h)][0] != val:
                flag = False
            if array[clamp(x+i,w)][clamp(y,h)][0] != val:
                flag = False
            if array[clamp(x,w)][clamp(y-i,h)][0] != val:
                flag = False
            if array[clamp(x,w)][clamp(y+i,h)][0] != val:
                flag = False
        return flag
        

    def updateGraph(self,FGmask):
        FG = []
        BG = []
        [w,h,null] = self.img.shape
        for i in range(w):
            for j in range(h):
                if FGmask[i][j][0] == 255:
                    tmp = []
                    tmp.append(self.img[i][j][0])
                    tmp.append(self.img[i][j][1])
                    tmp.append(self.img[i][j][2])
                    FG.append(tmp)
                else:
                    tmp = []
                    tmp.append(self.img[i][j][0])
                    tmp.append(self.img[i][j][1])
                    tmp.append(self.img[i][j][2])
                    BG.append(tmp)

        print('Training Model')
        FG, BG = np.array(FG), np.array(BG)
        FG_label = np.ones(FG.shape[0])
        BG_label = np.zeros(BG.shape[0])
        x_train = np.concatenate([FG, BG])
        y_train = np.concatenate([FG_label, BG_label])        
        #clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
        clf = svm.SVC(probability=True)
        clf.fit(x_train,y_train)
        #clf.fit(x_train)

        print('Adding New Edges')
        for i in range(0, w*h):
            # TODO: 1000 for seeds?
            #weights = -self.alpha * np.log(clf.predict_proba([self.Im[i]])[0])
            weights = -self.alpha * clf.predict_log_proba([self.Im[i]])[0]
            sourceWeight = weights[0]
            if sourceWeight > self.lam:
                sourceWeight = self.lam
            sinkWeight = weights[1]
            if sinkWeight > self.lam:
                sinkWeight = self.lam
            self.G.add_tedge(i, sourceWeight, sinkWeight)
            '''
            if self.is_surrounded(FGmask,i//w,i%w,255,step=3) == True:
                self.G.add_tedge(i, self.lam, 0)
            elif self.is_surrounded(FGmask,i//w,i%w,0,step=3) == True:
                self.G.add_tedge(i, 0, self.lam)
            else:
                weights = -self.alpha * np.log(clf.predict_proba([self.Im[i]])[0])
                sourceWeight = weights[0]
                if sourceWeight > self.lam:
                    sourceWeight = self.lam
                sinkWeight = weights[1]
                if sinkWeight > self.lam:
                    sinkWeight = self.lam
                self.G.add_tedge(i, sourceWeight, sinkWeight)
            '''
            '''
            if FGmask[i//w][i%w][0] == 255:
                self.G.add_tedge(i, self.lam, 0)
            else:
                self.G.add_tedge(i, 0, self.lam)
            '''


        print('Computing Mincut')
        self.G.maxflow()
        Iout = np.ones(shape = self.nodes.shape)
        for i in range(len(self.nodes)):
            Iout[i] = self.G.get_segment(self.nodes[i])

        return Iout

        
def clamp(x,X):
    if x < 0:
        x = 0
    elif x >= X:
        x = X-1
    return x

