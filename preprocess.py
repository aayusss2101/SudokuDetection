# Most Probably Useless

import cv2
import numpy as np

img=cv2.imread("sudoku.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,125,3)
#blur = cv2.GaussianBlur(gray,(7,7),0)
kernel=np.ones((3,3),np.uint8)
open  =cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
edges=cv2.Canny(open,100,200)
cv2.imshow("image",gray)
cv2.imshow("blur",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()