import numpy as np
import argparse
from cv2 import * #Import functions from OpenCV
import cv2
from PIL import Image


if __name__ == '__main__':
   # Load an color image in grayscale
 source = cv2.imread('C:\Users\Wala\Downloads\im.jpg',0) #0 for grayscale
# median filter
final = cv2.medianBlur(source, 9)
cv2.imwrite("median.bmp", final)
#min filter
kernel = np.ones((5,5), np.uint8)
img_erosion = cv2.erode(source, kernel, iterations=1)
cv2.imwrite("min.bmp",img_erosion)
#max filter
img_dilation = cv2.dilate(source, kernel, iterations=1)
cv2.imwrite("max.bmp",img_dilation)
#laplacian sopel - edge detector - second derivative
Laplacian = cv2.Laplacian(final, cv2.CV_8U, 3)
cv2.imwrite("Laplace.bmp",Laplacian)
#Since the Sobel kernels can be decomposed as the products
# of an averaging and a differentiation kernel, they compute the gradient with smoothing
#Vertical Sobel
sobely = cv2.Sobel(source,cv2.CV_64F,0,1,ksize=5) #convolved with a kernel to produce an
#approx of derivatives to get the vertical changes
# Y - RIGHT VISION
cv2.imwrite("Sobel_v.png",sobely)

#horizontal sobel
sobelx = cv2.Sobel(source,cv2.CV_64F,1,0,ksize=5)
# X= DOWN VISION
cv2.imwrite("Sobel_h.png",sobelx)




cv2.imshow('Original',source) #display
#resize the window
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original",400,400)

cv2.imshow('Median',final)
cv2.namedWindow("Median", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Median",400,400)

cv2.imshow('Max',img_dilation)
cv2.namedWindow("Max", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Max",400,400)
cv2.imshow('Min',img_erosion)
cv2.namedWindow("Min", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Min",400,400)
cv2.imshow('Laplace',Laplacian)
cv2.namedWindow("Laplace", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Laplace",400,400)
cv2.imshow('Vertical Sobel',sobely)
cv2.namedWindow("Vertical Sobel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vertical Sobel",400,400)
cv2.imshow('Horizontal Sobel',sobelx)
cv2.namedWindow("Horizontal Sobel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Horizontal Sobel",400,400)

cv2.waitKey(0)
cv2.destroyAllWindows()






