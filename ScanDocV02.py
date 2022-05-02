import time
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon
import skimage.filters.thresholding as threshold
import ExtendCV as exCV
import numpy as np
import imutils as im
import matplotlib.pyplot as plt
import itertools
import math
import cv2
import polygon_interacter as poly_i
import os
from pbtransform import pbt
import teseractOCR as tes
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from SRGAN.model import Generator
import argparse
from sys import argv
# from picamera.array import PiRGBArray
# from picamera import PiCamera
try:
    from pylsd.lsd import lsd
except ImportError:
    pass
class DocScanner: 
    def __init__(self, interactive = False, MIN_QUAD_AREA_RATIO = 0.25, MAX_QUAD_ANGLE_RANGE = 40):
        """
            Initialize the class
        """
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE
        
    def filter_corners(self, corners, min_dist=20):
        """Filters corners that are within min_dist of others"""
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                        for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners
    
    def angle_between_vectors_degrees(self, U, V):
        """
            Returns the angle in degrees between vectors 'U' and 'V'
        """
        return np.degrees((math.acos(np.dot(U, V) / (np.linalg.norm(U) * np.linalg.norm(V)))))
        
    def get_angle(self, p1, p2, p3):
        """
            Get the angle between three points
        """
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))
        
        avec = a - b
        cvec = c - b
        
        return self.angle_between_vectors_degrees(avec, cvec)
    
    def angle_range(self, quad):
        """
            Get the angle range of a quad
        """
        topLeft, topRight, bottomRight, bottomLeft = quad
        ura = self.get_angle(topLeft[0], topRight[0], bottomRight[0])
        ula = self.get_angle(bottomLeft[0], topLeft[0], topRight[0])
        lra = self.get_angle(topRight[0], bottomRight[0], bottomLeft[0])
        lla = self.get_angle(bottomRight[0], bottomLeft[0], topLeft[0])
        
        return np.ptp([ura, ula, lra, lla])
    
    def get_conner(self, img, lsd_implementation):
        """
            Get the list of corners in tuples like ((x, y), tuples) found in the image. With proper
            filtering and preprocessing, it should have at most 10 potential corners.
            
            Parameters:
                img: the image to be scanned (Walkthrough the preprocessing steps and filtering steps)
                lsd_implementation: the implementation of LSD to be used. Can be "pylsd" or "opencv"
        """
        lines = []
        if lsd_implementation == "pylsd":
            lines = lsd(img)
            
        if lsd_implementation == "opencv":    
            linesegmentdetected = cv2.createLineSegmentDetector(0)
            lines = linesegmentdetected.detect(img)[0]
        
        corners = []
        if lines is not None:
            # separate out the horizontal and vertical lines, and draw them back onto separate canvases
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                if lsd_implementation == "pylsd":
                    x1, y1, x2, y2, _ = line
                    
                if lsd_implementation == "opencv":    
                    x1, y1, x2, y2 = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            # find the horizontal lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # find the vertical lines (connected-components -> bounding boxes -> final lines)
            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # find the corners
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        # remove corners in close proximity
        corners = self.filter_corners(corners)
        return corners
    
    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        """
            Check if a contour is valid or not by checking all requirements are satisfied.
        """
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO
          and  self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)
        
    def get_contours(self, rescaled_image, lsd_implementation):
        """
            Returns a numpy array of shape (4, 2) containing the vertices of the four corners
            of the document in the image. It considers the corners returned from get_corners()
            and uses heuristics to choose the four corners that most likely represent
            the corners of the document. If no corners were found, or the four corners represent
            a quadrilateral that is too small or convex, it returns the original four corners.
        """
        # these constants are carefully chosen
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)

        # dilate helps to remove potential holes between edge segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # find edges and mark them in the output map using the Canny algorithm
        edged = cv2.Canny(dilated, 0, CANNY)
        test_corners = self.get_conner(edged, lsd_implementation)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = exCV.order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)

            # get top five quadrilaterals by area
            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
            # sort candidate quadrilaterals by their angle range, which helps remove outliers
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

            # for debugging: uncomment the code below to draw the corners and countour found 
            # by get_corners() and overlay it on the image

            # cv2.drawContours(rescaled_image, [approx], -1, (20, 20, 255), 2)
            # plt.scatter(*zip(*test_corners))
            # plt.imshow(rescaled_image)
            # plt.show()

        # also attempt to find contours directly from the edged image, which occasionally 
        # produces better results
        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        # loop over the contours
        for c in cnts:
            # approximate the contour
            approx = cv2.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        # If we did not find any valid contours, just use the whole image
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)
            
        return screenCnt.reshape(4, 2)
    
    def interactive_get_contour(self, screenCnt, rescaled_image):
        """
            Make the interactive with contours
        """
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots()
        ax.add_patch(poly)
        ax.set_title(('Drag the corners of the box to the corners of the document. \n'
            'Close the window when finished.'))
        p = poly_i.PolygonInteractor(ax, poly)
        plt.imshow(rescaled_image)
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype = "int32")
        return new_points.reshape(4, 2)

    def scan(self, image_path, lsd_implementation):
        """
            Scan the document combine all the steps
        """
        RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = 'output'

        # load the image and compute the ratio of the old height
        # to the new height, clone it, and resize it
        image = cv2.imread(image_path)

        assert(image is not None)

        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = im.resize(image, height = int(RESCALED_HEIGHT))
        
        # effect the britness of the image
        rescaled_image = self.ajustBrightnessContrast(rescaled_image)

        # get the contour of the document
        screenCnt = self.get_contours(rescaled_image, lsd_implementation)

        if self.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        # apply the perspective transformation
        warped = exCV.four_point_transform(orig, screenCnt * ratio)

        # convert the warped image to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # sharpen image
        sharpen = cv2.GaussianBlur(gray, (0,0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
        sharpen = cv2.resize(sharpen, (warped.shape[1], warped.shape[0]))
        
        # apply adaptive threshold to get black and white effect
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        # save the image
        basename = os.path.basename(image_path)
        cv2.imwrite(".\Images\\adaptiveThreshold\\" + basename, thresh)
        
        # show the transformed image
        stack = [[rescaled_image, warped, sharpen, thresh]]
        label = [["Original", "warped", "Sharpen", "Threshold"]]
        stackImg = exCV.stackImages(0.75, stack, label)
        
        #See WorkFlow for the image processing
        cv2.imshow("WorkFlow", stackImg)
        
        while True:        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow("WorkFlow")
                text = self.OCR(thresh)
                print(text)
                f = open("output/output" + basename.replace(".", "") + ".txt", mode = "w+", encoding = "utf-8")
                for word in text.split(" "):
                    f.write(word + " ")
                f.close()
                break
            if cv2.waitKey(1) & 0xFF == ord('u'):
                cv2.destroyWindow("WorkFlow")
                weight = thresh.shape[1]
                height = thresh.shape[0]
                imgUp = thresh
                if (weight < 700 and height < 1000):
                    imgUp = cv2.imread(".\Images\\adaptiveThreshold\\" + basename)
                    imgUp = cv2.merge([imgUp[:,:,0], imgUp[:,:,1], imgUp[:,:,2]])
                    imgUp = self.upResolution(imgUp, ".\Images\\adaptiveThreshold\\" + basename)
                text = self.OCR(imgUp)
                print(text)
                f = open("output/output" + basename.replace(".", "") + ".txt", mode = "w+", encoding = "utf-8")
                for word in text.split(" "):
                    f.write(word + " ")
                f.close()
                break
            
        
    def ajustBrightnessContrast(self, image):
        """
            Adjust the brightness and contrast of the image
        """
        # initialize the valtrackbar for the contrast and brightness
        pbt.initialize_Trackbars_BC()
        while True:        
            # Get the valtrackbar value
            brightness, contrast = pbt.BrightnessContrast_Effect(0)
            # Apply the brightness and contrast effect
            output = pbt.controller(image, brightness=brightness, contrast=contrast)
            # Show the image
            cv2.imshow("Effect_BC", output)
            # Option to apply effect on image
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.destroyWindow("Effect_BC")
                cv2.destroyWindow("BrightnessContrast")
                return output
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow("Effect_BC")
                cv2.destroyWindow("BrightnessContrast")
                break
        return image
    
    def OCR(self, image):
        """
            Apply OCR to the image
        """
        output_ocr = tes.get_string(image)
        return output_ocr

    def upResolution(self, image_adaptive_threshold, path):
        """
            Up the resolution of the image using model SRGAN
        """
        img = image_adaptive_threshold
        model = Generator(4).eval()
        model.load_state_dict(torch.load('SRGAN/epochs/' + 'netG_epoch_4_50.pth', map_location=lambda storage, loc: storage))
        img = Variable(ToTensor()(img), volatile=True).unsqueeze(0)
        out = model(img)
        out_img = ToPILImage()(out[0].data.cpu())
        basename = os.path.basename(path)
        out_img.save('./Images/upRe/out_srf_' + str(4) + '_' + basename)
        return out_img
    
                
parser = argparse.ArgumentParser(description = "Get text from image files")
action_choices = ["single", "realtime"]
parser.add_argument("-m" ,"--mode", help = "Scan on single image or realtime camera", choices = action_choices, default = "realtime")
parser.add_argument("-t", "--type", help = "Scan realtime camera by piCamera or normal camera", choices = ["pi", "normal"], default = "normal", required = (action_choices[1] in argv))
parser.add_argument("-i" ,"--image", help = "Path to image file", required=(action_choices[0] in argv))
parser.add_argument("-lsd", "--line_segmentation_detection", help = "Use line segmentation detection", default = "opencv", choices = ["opencv", "pylsd"], required = True)
opt = parser.parse_args()

MODE = opt.mode
IMAGE = opt.image
TYPE = opt.type
LSD = opt.line_segmentation_detection

if MODE == "single":
    Scanner = DocScanner(interactive = True)
    Scanner.scan(IMAGE, LSD)
    cv2.waitKey(0)
if MODE == "realtime" and TYPE == "normal":
    cap = cv2.VideoCapture(0)
    cap.set(10 , 150)
    time.sleep(2)

    while True:
        _, frame = cap.read()
        cv2.imshow("RealTimeScan", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('w'):
            get_dir = os.listdir("./Images/Scanned")
            numberofFiles = len(get_dir)
            numberofFiles += 1
            cv2.imwrite("./Images/Scanned/output%d.jpg" %numberofFiles, frame)
            Scanner = DocScanner(interactive=True)
            Scanner.scan("./Images/Scanned/output%d.jpg" %numberofFiles, LSD)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if MODE == "realtime" and TYPE == "pi":
    pass
    # # Initialize the camera
    # camera = PiCamera()
    
    # # Set the camera resolution
    # camera.resolution = (640, 480)
    
    # # Set the number of frames per second
    # camera.framerate = 32
    
    # # Generates a 3D RGB array and stores it in rawCapture
    # raw_capture = PiRGBArray(camera, size=(640, 480))
    
    # # Wait a certain number of seconds to allow the camera time to warmup
    
    # # Capture frames continuously from the camera
    # for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    #     cv2.imshow("RealTimeScan", frame.array)
        
    #     raw_capture.truncate(0)
        
    #     if cv2.waitKey(1) & 0xFF == ord('w'):
    #         get_dir = os.listdir("Scanned")
    #         numberofFiles = len(get_dir)
    #         numberofFiles += 1
    #         cv2.imwrite("./Images/Scanned/output%d.jpg" %numberofFiles, frame)
    #         Scanner = DocScanner(interactive=True)
    #         Scanner.scan("./Images/Scanned/output%d.jpg" %numberofFiles, LSD)
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break