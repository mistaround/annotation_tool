import numpy as np
import maxflow
from sklearn import mixture 
import cv2

class GCGraph(object):
    def __init__(self,img):
        self.img = np.array(img)
        self.Im = self.img.reshape(-1,3)

        self.k = 1
        self.sigma = 1
        self.lam = 4
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
        x_train = np.concatenate([FG, BG])        
        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
        clf.fit(x_train)

        print('Adding New Edges')
        for i in range(0, w*h):
            weights = self.lam * clf.predict_proba([self.Im[i]])
            sourceWeight = weights[0][0]
            sinkWeight = weights[0][1]
            # TODO: 1000 for seeds?
            self.G.add_tedge(i, sourceWeight, sinkWeight)

        print('Computing Mincut')
        self.G.maxflow()
        Iout = np.ones(shape = self.nodes.shape)
        for i in range(len(self.nodes)):
            Iout[i] = self.G.get_segment(self.nodes[i])

        return Iout

        


