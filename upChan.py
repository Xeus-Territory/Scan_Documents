import cv2
import numpy as np

def upchange(image_path):
    """
        Update channel of image => Create input for model SRGAN
    """
    img = cv2.imread(image_path)
    width = img.shape[1]
    height = img.shape[0]
    img = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2]])
    print(img.shape)
