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
#pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def post_process(str):
    """
        Post process text for reduce some text and error handling
    """
    # process for the strange character
    strangeCha = "=+*^#@!~`_+{}[]|\\<>/~—¬„Ø()\'\":;-“”%"
    #strangeCha = ["-.", "-,", "-=", "-*", "-^", "-#", "-@", "-!", "-~", "-`", "-_", "-{", "-}", "-[", "-]", "-|", "-\\", "-<", "->", "-/", "-~", "-—", "-¬", "-„", "-“", "-”"]  
    for cha in strangeCha:
        str = str.replace(cha, "    ")
    
    # process for the syntax error
    syntax = [". :"]
    for s in syntax:
        if s == ". :":
            str = str.replace(s, ".")
            
    # process for the space
    str = str.strip()
    str = str.replace("  ", "")
    str = " ".join(str.split())
    
    
    # process for grammar error   
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sign = ",;:!?()[]{}<>/\"'`&+-=*^#@!~`_+{}[]|\\"
    alphabet = " ".join(alphabet)
    sign = " ".join(sign)
    grammar = alphabet + " " + sign
    for g in grammar:
        str = str.replace(" . " + g + ". ", "  ")
        str = str.replace(" " + g + " ", "  ")
        str = str.replace("." + g, "  ")
        
        
    # process for the number and date 
    str = normalize_text(str)
    
    return str

def get_string(img):
    """
        Get string from image
    """
    result = pytesseract.image_to_string(img, lang = "vie")
    result = post_process(result)
    cv2.waitKey(0)
    return result