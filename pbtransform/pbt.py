import numpy as np
import cv2

## Define the trackbars to adjust the brightness and contrast
def initialize_Trackbars_BC():
    cv2.namedWindow("BrightnessContrast")
    cv2.resizeWindow("BrightnessContrast", 360, 240)
    cv2.createTrackbar("Brightness", "BrightnessContrast", 255, 2 * 255, BrightnessContrast_Effect)
    cv2.createTrackbar("Contrast", "BrightnessContrast", 127, 2 * 127, BrightnessContrast_Effect)

# Create the trackbars into 1 frame in OS
def BrightnessContrast_Effect(err = 0):
    brightness = cv2.getTrackbarPos("Brightness", "BrightnessContrast")
    contrast = cv2.getTrackbarPos("Contrast", "BrightnessContrast")
    
    return brightness, contrast
    
# Change the brightness and contrast of the image
def controller(img, brightness = 255, contrast = 127):
    
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
    
        B_alpha = (max - shadow) /255
        B_gamma = shadow
        
        cal = cv2.addWeighted(img, B_alpha, img, 0, B_gamma)
        
    else:
        cal = img
        
    if contrast != 0:
        C_alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        C_gamma = 127 * (1- C_alpha)
        
        cal = cv2.addWeighted(cal, C_alpha, cal, 0 , C_gamma)
    
    # Check detail of info new image
    # cv2.putText(cal, f'Brightness: {brightness}  ,Contrast: {contrast}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return cal

# Test performance of function
# if __name__ == '__main__':
#     img = cv2.imread("D:\Code\Project\PBL5_KTMT\Source\PBT\\334698.jpg")
    
#     initialize_Trackbars_BC()
    
#     while True:
#         brightness, contrast = BrightnessContrast_Effect(0)
        
#         output = controller(img, brightness, contrast)
        
#         cv2.imshow("Effect", output)        
          
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
        
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             cv2.imwrite("D:\Code\Project\PBL5_KTMT\Source\PBT\Resolution\output.jpg", output)
        