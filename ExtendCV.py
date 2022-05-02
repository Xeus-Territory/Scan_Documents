import cv2
import numpy as np
from scipy.spatial import distance as dist

# Refference to github:
# https://github.com/andrewdcampbell/OpenCV-Document-Scanner/blob/master/pyimagesearch/transform.py


def order_points(pts):
    """
        Given a 4 point polygon, it will return the ordered points
    """
    xSorted = pts[np.argsort(pts[:, 0]), :] # Sort points according x_axis
    
    # Grap the left-most and right-most points from the sorted x-axis
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    #Sort the left-most points according to their y-axis => top-left, bottom-left
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (topLeft, bottomLeft) = leftMost
    
    # Use to top-left point to find the euclid distance between it and right-most
    # the point with the largers distance will be the bottom-right point
    Edist = dist.cdist(topLeft[np.newaxis], rightMost, "euclidean")[0]
    (bottomRight, topRight) = rightMost[np.argsort(Edist)[::-1], :]
    
    print(topLeft, topRight, bottomRight, bottomLeft)
    
    return np.array([topLeft, topRight, bottomRight, bottomLeft], dtype = "float32")

def four_point_transform(img, pts):
    """
        Given a 4 point polygon, it will return the transformed image
    """
    rect = order_points(pts)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    
    # compute the width of the new image will be the maximum of
    # distance between bottom-right and bottom-left of x-axis
    # or distance between top-right and top-left of x-axis
    widthA = np.sqrt((bottomRight[0] - bottomLeft[0])**2 + (bottomRight[1] - bottomLeft[1])**2)
    widthB = np.sqrt((topRight[0] - topLeft[0])**2 + (topRight[1] - topLeft[1])**2)
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image will be the maximum of
    # distance between bottom-right and top-right of y-axis
    # or distance between top-left and bottom-left of y-axis
    heightA = np.sqrt((topRight[0] - bottomRight[0])**2 + (topRight[1] - bottomRight[1])**2)
    heightB = np.sqrt((topLeft[0] - bottomLeft[0])**2 + (topLeft[1] - bottomLeft[1])**2)
    maxHeight = max(int(heightA), int(heightB))
    
    # have dim of the new image ==> construct set of destination points to obtain
    # a "birds eye view" of the image, again specifying points in the top-left, bottom-left,
    # and bottom-right, topRight
    destination = np.array([
        [0, 0], 
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype = "float32")
    
    matrixPerspective = cv2.getPerspectiveTransform(rect, destination)
    warpImg = cv2.warpPerspective(img, matrixPerspective, (maxWidth, maxHeight))
    
    return warpImg

def stackImages(scale,imgArray, label = []):
    """
        Stacks images horizontally
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor_con)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(label) > 0:
        widthImg = int(ver.shape[1] / cols)
        heightImg = int(ver.shape[0] / rows)
        for x in range(0, rows):
            for y in range(0, cols):
                cv2.rectangle(ver, (y * widthImg, x * heightImg), ((y * widthImg + len(label[x][y]) * 13 + 27, x * heightImg + 30)), (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, label[x][y], (widthImg*y + 10, height * x + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
    return ver

def reOrder(Points):
    """
        Return the points in the correct order
    """
    Points = Points.reshape((4, 2))
    NewPoints = np.zeros((4, 1, 2), dtype = np.int64)
    addPoint = Points.sum(1)
    NewPoints[0] = Points[np.argmin(addPoint)]
    NewPoints[3] = Points[np.argmax(addPoint)]
    diff = np.diff(Points, axis = 1)
    NewPoints[1] = Points[np.argmin(diff)]
    NewPoints[2] = Points[np.argmax(diff)]
    return NewPoints
  
def getContours(img):
    """
        Return the contours of the image
    """
    biggest = np.array([])
    maxArea = 0
    contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                
    return biggest, maxArea

def drawRectangle(img, biggest, thickness):
    """
        Draw the rectangle on the image
    """
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img

def nothing(x):
    pass

def initializeTrackbars(intialTracbarVals=0):
    """
        Initialize the trackbars for threshold
    """
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 200,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 200, 255, nothing)


def valTrackbars():
    """
        Return the values of the trackbars
    """
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src
