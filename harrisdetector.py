import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List
import math

def harris_corners(img: np.ndarray, threshold=1.0, blur_sigma=2.0) -> List[Tuple[float, np.ndarray]]:

    dx = cv2.Sobel(img,3,1,0,img)
    dy = cv2.Sobel(img,3,0,1,img)
    """
    print("dx shape: " + str(dx.shape))
    print("dy shape: " + str(dy.shape))
    print("dx dy: " + str(dx*dy))
    print("dx dx shape: " + str((dx*dx).shape))
    print("dx dy shape: " + str((dx*dy).shape))
    print("dy dy shape: " + str((dy*dy).shape))
    """
    M = np.array([[dx*dx, dx*dy],[dx*dy, dy*dy]])
    det1_M = M[0,1]*M[1,0]
    det_M = M[0,0]*M[1,1]-M[0,1]*M[1,0]
    trace_M = M[0,0] + M[1,1]
    R = det_M/trace_M
    print(np.amax(R))

    """
    R = det_M - 0.05*(trace_M*trace_M)
    R = R-np.amin(R)
    print(R)
    R = R/np.amax(R)
    print(R)
    R = R*255
    print(R)
    R = -R
    print(R)
    R = R-np.amin(R)
    print(R)
    """

    """
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if abs(R[i-1][j-1]) > threshold:
                print("i: " + str(i) + "j: " + str(j))
                print(R[i-1][j-1])
    """
    
    return R

    """
    Return the harris corners detected in the image.
    :param img: The grayscale image.
    :param threshold: The harris respnse function threshold.
    :param blur_sigma: Sigma value for the image bluring.
    :return: A sorted list of tuples containing response value and image position.
    The list is sorted from largest to smallest response value.
    """
    raise NotImplementedError
