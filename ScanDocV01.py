import numpy as np
import cv2
import ExtendCV as exCV

height = 640
width = 480
cap = cv2.VideoCapture(0)
cap.set(10, 150)
    
exCV.initializeTrackbars()
count = 0

def scan():
    """
        Realtime detection document | basic method    
    """
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (width, height))
        imgBlank = np.zeros((height, width, 3), np.uint8)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        thres = exCV.valTrackbars()
        imgCanny = cv2.Canny(imgBlur, thres[0], thres[1])
        #### Remove the shadown different from the doc ####
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
        imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
        
        ## FIND ALL CONTOURS
        imgContours = img.copy()
        imgBigContours = img.copy()
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
        
        ## FIND BIGGEST CONTOURS
        biggest, maxArea = exCV.getContours(imgThreshold)
        if biggest.size != 0:
            biggest = exCV.reOrder(biggest)
            cv2.drawContours(imgBigContours, biggest, -1, (0, 255, 0), 10)
            imgBigContours = exCV.drawRectangle(imgBigContours, biggest, 2)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWrapColored = cv2.warpPerspective(img, matrix, (width, height))
            
            ####Remove 20 Pixels from each side
            imgWrapColored = imgWrapColored[20:imgWrapColored.shape[0]-20, 20:imgWrapColored.shape[1]-20]
            imgWrapColored = cv2.resize(imgWrapColored, (width, height))
            
            #### Apply adaptive threshold
            imgWrapGray = cv2.cvtColor(imgWrapColored, cv2.COLOR_BGR2GRAY)
            imgAdaptiveThreshold = cv2.adaptiveThreshold(imgWrapGray, 255, 1, 1, 7 , 2)
            imgAdaptiveThreshold = cv2.bitwise_not(imgAdaptiveThreshold)
            imgAdaptiveThreshold = cv2.medianBlur(imgAdaptiveThreshold, 3)
            
            ### Image Array for Display
            imageArray = ([img, imgGray, imgThreshold, imgContours ],
                        [imgBigContours, imgWrapColored, imgWrapGray, imgAdaptiveThreshold])
        else :
            imageArray = ([img, imgGray, imgThreshold, imgContours ],
                        [imgBlank, imgBlank, imgBlank, imgBlank])
            
        # LABELS FOR DISPLAY
        labels = [["Original", "Gray", "Threshold", "Contours"],
                ["Biggest Contours", "Warp Prespective", "Wrap Gray", "Adaptive Threshold"]]
        
        stackImages__ = exCV.stackImages(0.75, imageArray, labels)
        cv2.imshow("Result", stackImages__)
        
        # SAVE IMAGE WHEN 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("Scanned/myImage"+str(count)+".jpg",stackImages__)
            cv2.rectangle(stackImages__, ((int(stackImages__.shape[1] / 2) - 230), int(stackImages__.shape[0] / 2) + 50),
                        (1100, 350), (0, 255, 0), cv2.FILLED)
            cv2.putText(stackImages__, "Scan Saved", (int(stackImages__.shape[1] / 2) - 200, int(stackImages__.shape[0] / 2)),
                        cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.imshow('Result', stackImages__)
            cv2.waitKey(300)
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()