
import math
import cv2
import numpy as np
from scipy.sparse.extract import find
from sklearn.cluster import KMeans

def dominant_using_bincount(a):
    print(a.shape[-1])
    a2D = a.reshape(-1,a.shape[-1])
    print(a2D.shape)
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)
    

def dominant_using_kmeans(img):
    #img=cv2.imread(img_path)
    img=img/255
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    print(img.shape)
    kmeans = KMeans(n_clusters=3,max_iter=30)
    kmeans.fit(img)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)
      #the cluster centers are our dominant colors.
    cvt_bgr = tuple(map(int,255*(kmeans.cluster_centers_[np.bincount(kmeans.labels_).argmax()][:])))
    return cvt_bgr


def main():
    
    img=cv2.imread("/home/saurav/Downloads/hardhat6.jpg")
    
    print(dominant_using_bincount(img))
    print(dominant_using_kmeans(img))
    
    
    
    

if __name__== '__main__':
    main()


