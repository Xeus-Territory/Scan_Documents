# PROJECT ACKNOWLEDGEMENTS ABOUT SCAN DOCUMENTATION WITH GITHUB API
---
## First, I want to thank 
- [GitHub](https://github.com) for the API
- [Murtaza Hassan](https://github.com/murtazahassan) for the tutorial in OpenCV
- [tesseract-ocr](https://github.com/tesseract-ocr) for the OCR engine
- [andrewdcampbell](https://github.com/andrewdcampbell) for the idea of using the OpenCV library to do the image processing
- [leftthomas](https://github.com/leftthomas) for the SRGAN model
<pre> &#8594 Combined, these people have made this project possible, I appreciate their work and contribution and I happy to do this project.</pre>
---

## PROBLEM SOLVING: 
### Ideal Part: Image processing to convert image text into Vietnamese language, centralized system into handwriting recognition, raw image processing, and model quality improvement SRGAN and extract the data using OCR. The results obtained from the group were successful in the conversion and have successfully used the application results for the processing sound
### THE IMAGE PROCESSING PART (INCLUDING ALL STEPS):
![Target Solution](Images/Solution/ProcessImageAndOCR.drawio.png)
## <center> Target Solution </center> <br>
![Image Processing](Images/Solution/StandardModel(ImageToText).drawio.png)
## <center> Image Processing </center> <br>
- Process capture and scan images for handwriting recognition and process raw images captured by image transformation.
- Pre-processing of raw images by enabling image enhancement with SRGAN to improve image quality without losing image detail.
- Extract text using OCR – specifically the Tesseract engine through the
available training data.

## Go through the details of the image processing part:
[1] Capture and scan images for handwriting recognition or printed text and process raw images captured by image transformation.
---
![Raw Processing](Images/Solution/Screenshot%202022-06-07%20145556.png)
## Result of Raw Processing Part:
![Result Raw](Images/Solution/ResultRaw.png)<br>
[2] Pre-processing of raw images by enabling image enhancement with SRGAN to improve image quality without losing image detail.
***Notice*** That image not 100% improved because the Type of image is not the same as the training data (So Carefully) and one thing is that model using much CPU time and Memory so it need much time for processing.
---
![SRGAN](Images/Solution/Screenshot%202022-06-07%20145556.png)
## Result of SRGAN Processing Part:
![ResultSRGAN](Images/Solution/ResultPart2.png)<br>
[3] Extract text using OCR – specifically the Tesseract engine through the available training data.
---
![OCR](Images/Solution/Screenshot%202022-06-07%20145608.png)<br>
## Result of OCR Processing Part:
![OCR](Images/Solution/ResultPart3.png)<br>
[4] The Final Step is sending the result in step 3 for server to process and return the result in style URL and get this URL and play this text we converted.
Have Good Team to do this other part i want to say thanks for colap with them [congtoan](https://github.com/toanil315), [trian](https://github.com/triandn) and [thanhtue](https://github.com/TueTran1). The detail of the step convert text to audio can visit [github](https://github.com/toanil315/flask-tacotron2) and the step send and receive the result from server can visit [github](https://github.com/Xeus-Territory/Github_API_Requests) for details
---
    - So I have to say good experience with this AI project.

### So let run code and see what you get:
![](Images/Solution/Result%20all.png)
<pre>python ScanDocV02.py -lsd opencv</pre>
---

## Run the code and enjoy the result. :coffee:
