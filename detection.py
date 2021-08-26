import numpy as np
import cv2
from tensorflow import keras
from sudokuAlgorithm import *
import os

# Mutes warnings of Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def processImage(img):

    '''
    Preprocesses Image
    
    Parameters:
        img (numpy.ndarray) : Input Image

    Returns:
        dilate (numpy.ndarray) : Processed Image

    '''
    
    # Converting img to Grayscale
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian Blur on gray using a 11*11 Kernel
    blur=cv2.GaussianBlur(gray, (11, 11), 0)


    # Applying Adaptive Threshold on blur with maxVal=255, adaptive method=Adaptive Thresh Mean Constant, thresholdType=Threshold Binary Inverse, blockSize=5 and constant=2
    thresh=cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)

    # Kernel for Dilating Operation
    kernel=(6, 6)
    # Applying Dilate on thresh using kernel
    dilate=cv2.dilate(thresh, kernel)

    return dilate


def findMaxContour(processedImg):

    '''
    Finds Contour with Maximum Area

    Parameters:
        processedImg (numpy.ndarray) : Input Image

    Returns:
        maxCnt (numpy.ndarray) : Contour having Maximum Area
    
    '''

    # List of Contours of processedImg
    contours, _ =cv2.findContours(processedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Placeholder variable to store the area of contours
    maxArea=0
    # Stores the Contour having maximum area
    maxCnt=None

    for cnt in contours:

        # Calculates area of contour
        area=cv2.contourArea(cnt)

        if area>maxArea:
            maxCnt=cnt
            maxArea=area

    return maxCnt


def orderPoints(approx):

    '''
    Order Points of approx in the form [Upper Left, Upper Right, Lower Left, Lower Right]

    Parameters:
        approx (numpy.ndarray) : Approximate Polynomial Curve
    
    Returns:
        np.array: Points in ordered form

    '''

    one=approx[np.argmin(list(map(lambda x: x[0]+x[1], approx)))]
    two=approx[np.argmax(list(map(lambda x: x[0]-x[1], approx)))]
    three=approx[np.argmin(list(map(lambda x: x[0]-x[1], approx)))]
    four=approx[np.argmax(list(map(lambda x: x[0]+x[1], approx)))]
    return np.array([one,two,three,four])



def transformImage(maxCnt, processedImg):

    '''
    Transforms processedImg into a 252*252 sized Perspective Transformed Image
    
    Parameters:
        maxCnt (numpy.ndarray) : Contour having Maximum Area
        processedImg (numpy.ndarray) : Input Image

    Returns:
        approx (numpy.ndarray) : Approximate Polynomial Curve for maxCnt
        dst (numpy.ndarray) : Destination Curve for Perspective Transform
        warp (numpy.ndarray) : Perspective Transformed Image

    '''

    # Approximates a polynomial curve
    approx=cv2.approxPolyDP(maxCnt, 0.01*cv2.arcLength(maxCnt, True), True).reshape(4,2)
    approx=np.array(approx, dtype='float32')
    approx=orderPoints(approx)

    dst=np.array([[0,0],[252,0],[0,252],[252,252]], dtype='float32')

    # Calculates Matrix for Perspective Transform from approx to dst
    M=cv2.getPerspectiveTransform(approx, dst)
    # Applying Warp Perspective on processedImg using M
    warp=cv2.warpPerspective(processedImg,M,(252, 252))

    return approx,dst,warp


def getSudokuMask(processedImg,model,size=28):

    '''
    Finds Digits in Sudoku Image
    
    Parameters:
        processedImg (numpy.ndarray) : Processed Image
        model (tf.keras.Model) : ML Model used to Detect Digits
        size (Int) : Side length of Square Grid (Default Value=28)

    Returns:
        sudoku (numpy.ndarray) : 2D Array containing digits in Sudoku Image
        mask (numpy.ndarray) : 2D Array containing digits in Sudoku Image

    '''
    
    sudoku=[]   
    mask=[] 

    for i in range(9):
        for j in range(9):
            
            # End Points for size*size Grid
            xmin=i*size
            xmax=xmin+size
            ymin=j*size
            ymax=ymin+size

            # Cropping Grid from processedImg and processing it
            temp=processedImg[xmin:xmax, ymin:ymax]
            temp=255-temp
            temp=temp.reshape((1, size, size, 1))

            # Getting Predicted Digit
            y_temp=model.predict(temp)
            foo=np.argmax(y_temp, axis=-1)

            sudoku.append(foo)
            mask.append(foo)
            #print(foo,end=' ')
        #print()

    sudoku=np.array(sudoku)
    sudoku=sudoku.reshape((9,9))
    mask=np.array(mask)
    mask=mask.reshape((9,9))

    return sudoku, mask


def drawOnSudoku(sudoku, mask, warp):

    '''
    Draws solution on processed image if possible

    Parameters:
        sudoku (numpy.ndarray) : 2D Array containing digits in Sudoku Image
        mask (numpy.ndarray) : 2D Array containing digits in Sudoku Image
        warp (numpy.ndarray) : Processed Input Image
    
    Returns:
        Boolean : True if sudoku was solvable else False
        solutionImg (numpy.ndarray) : Image containing solved Sudoku

    '''

    solutionImg=255-warp
    #showImage(solutionImg)

    # Solving Sudoku if possible
    if solveSudoku(sudoku):

        for r in range(9):
            for c in range(9):
                
                if mask[r][c]!=0:
                    continue

                # Puts Text on solutionImg for grids where mask[r][c] is zero
                cv2.putText(solutionImg, str(sudoku[r][c]), (28*(c)+3, 28*(r+1)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0))
        #showImage(solutionImg)
        return True, solutionImg
    else:
        return False, None



def reverseTransform(approx, dst, sol, processedImg):

    '''
    Reverse Transforms processedImg from a 252*252 sized Perspective Transformed Image
    
    Parameters:
        approx (numpy.ndarray) : Approximate Polynomial Curve
        dst (numpy.ndarray) : Source Curve for Perspective Transformation
        sol (numpy.ndarray) : Solution Image
        processedImg (numpy.ndarray) : Processed Sudoku Image
    Returns:
        res (numpy.ndarray) : Processed Image

    '''

    # Calculates Matrix for Perspective Transform from dst to approx
    M=cv2.getPerspectiveTransform(dst, approx)
    # Applying Warp Perspective on warp
    rev=cv2.warpPerspective(sol, M, (processedImg.shape[1], processedImg.shape[0]))

    return rev


def maskImages(img, mask):

    '''
    Overlays mask on img
    
    Parameters:
        img (numpy.ndarray) : Base Image
        mask (numpy.ndarray) : Mask Image

    Returns:
        res (numpy.ndarray) : Resultant Image
    
    '''

    _, alpha=cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # Converting mask from Grayscale to BGR
    mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Extracting Individual Colour Channels
    b, g, r=cv2.split(mask)
    rgba=[b, g, r, alpha]
    # Merging Channels along with alpha
    mask=cv2.merge(rgba, 4)


    alpha=255*np.ones(img.shape[:2],np.uint8)
    # Extracting Individual Colour Channels
    b, g, r=cv2.split(img)
    rgba=[b, g, r, alpha]
    # Merging Channels along with alpha
    img=cv2.merge(rgba, 4)

    # Doing Bitwise AND of img and mask
    res=cv2.bitwise_and(img, mask)

    return res


def sudokuDetection(path):

    '''
    Finds solution to Sudoku
    
    Parameters:
        path (String) : Path of Sudoku Image

    Returns:
        Boolean : True if solution to sudoku was found else False
        res : Image with solution to sudoku if solution was possible else None
    
    '''

    try:

        # Original Sudoku Image
        img=cv2.imread(path)

        processedImg=processImage(img)
        
        # Contour with Max Area    
        maxCnt=findMaxContour(processedImg)

        approx, dst, warp=transformImage(maxCnt,processedImg)
        
        # Load Model
        model_path="/home/aayussss2101/Desktop/SudokuDetection/digitModel"
        model=keras.models.load_model(model_path)

        sudoku, mask=getSudokuMask(warp,model)

        success, sol=drawOnSudoku(sudoku,mask,warp)

        if(success):
            rev=reverseTransform(approx,dst,sol,processedImg)
            res=maskImages(img, rev)
            return True, res

    except Exception as e:
        print(e)
        
    return False, None