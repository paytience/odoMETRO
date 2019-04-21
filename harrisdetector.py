import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List
import math

def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:

    dx = cv2.Sobel(img,3,1,0,img)
    dy = cv2.Sobel(img,3,0,1,img)
    M = np.array([[dx*dx, dx*dy],[dx*dy, dy*dy]])
    sumM = np.zeros(M.shape)
    for x in [0,1]:
        for y in [0,1]:
            sumM[x,y] = cv2.filter2D(M[x,y],-1,np.ones([3,3]));

    det_M = sumM[0,0]*sumM[1,1]-sumM[0,1]*sumM[1,0]
    trace_M = sumM[0,0] + sumM[1,1]
    R = det_M - 0.04*trace_M*trace_M

    filterR = np.array(R);
    filterR[filterR<threshold] = 0;

    window_size = 7;
    H,W = filterR.shape;
    for x in range(H-window_size+1):
        for y in range(W-window_size+1):
            window = filterR[x:x+window_size, y:y+window_size]
            window[window<np.amax(window)] = 0
    
    dst = cv2.dilate(R,None)
    maxPoints  = []

    for i in range(50):
        pos = np.unravel_index(np.argmax(filterR),filterR.shape);
        maxPoints.append((filterR[pos],pos[::-1]))
        filterR[pos] = 0
    
    return filterR, maxPoints

    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    raise NotImplementedError
