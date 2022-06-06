from turtle import st
import cv2
from matplotlib import image
import pytesseract
from PIL import Image
import numpy as np
import os
from format_text import normalize_text

"""
    Tesseract OCR for get_string from image
"""

# Get the path tesseract-ocr if we place it in the same folder
#pytesseract.pytesseract.tesseract_cmd = r".\Tesseract-OCR\\tesseract.exe"

# Get the path tesseract-ocr if we place it in default directory
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def post_process(str):
    """
        Post process text for reduce some text and error handling
    """
    try:
        
        # process for the strange character
        strangeCha = "=+*^#@!~`_+{}[]|\\<>/~—¬„Ø()\'\":;-“”%ø?äˆø‡$¡"
        #strangeCha = ["-.", "-,", "-=", "-*", "-^", "-#", "-@", "-!", "-~", "-`", "-_", "-{", "-}", "-[", "-]", "-|", "-\\", "-<", "->", "-/", "-~", "-—", "-¬", "-„", "-“", "-”"]  
        for cha in strangeCha:
            str = str.replace(cha, "  ")
        
        # process for the syntax error
        syntax = [". :", "..", "..."]
        for s in syntax:
            if s == ". :":
                str = str.replace(s, ".")
            if s == ".." or s == "...":
                str = str.replace(s, ".")
                
        # process for the space
        str = str.strip()
        str = str.replace("    ", "")
        str = " ".join(str.split())
        
        # process for last word in text
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if str[-1] not in alphabet:
            str = str.replace(str[-1], "")
            
        # process for grammar error   
        # alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # sign = ",;:!?()[]{}<>/\"'`&+-=*^#@!~`_+{}[]|\\"
        # alphabet = " ".join(alphabet)
        # sign = " ".join(sign)
        # grammar = alphabet + " " + sign
        # for g in grammar:
        #     str = str.replace(" . " + g + ". ", "  ")
        #     str = str.replace(" " + g + " ", "  ")
        #     str = str.replace("." + g, "  ")
            
        # process for the number and date 
        str = normalize_text(str)
    except:
        print("No text detected")
        str = ""
        
    return str

def increseAccuracy(img):
    """
        Increase accuracy of tesseract by split image 2 part and OCR in 2 part | decrease time but not increase accuracy
    """
    # increase accuracy
    (h_thr, w_thr) = img.shape[:2]
    s_idx = 0
    e_idx = int(h_thr / 2)
    last_re = ""
    for _ in range(0, 2):
        crp = img[s_idx:e_idx, 0:w_thr]
        (h_crp, w_crp) = crp.shape[:2]
        crp = cv2.resize(crp, (w_crp*2, h_crp*2))
        crp = cv2.erode(crp, None, iterations=1)
        s_idx = e_idx
        e_idx = s_idx + int(h_thr / 2)
        result = pytesseract.image_to_string(crp, lang = "vie")
        result = post_process(result)
        last_re = last_re + " " + result
        cv2.waitKey(0)
    return last_re

def get_string(img):
    """
        Get string from image
    """
    result = pytesseract.image_to_string(img, lang = "vie")
    result = post_process(result)
    cv2.waitKey(0)
    return result