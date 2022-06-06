"""
    Setting up some methods for the main program for onces of model OS
"""

##### Method for single image using take single image
# Scanner = DocScanner(interactive = True)
# Scanner.scan('.\Images\Image_Input\\5.jpg')
# cv2.waitKey(0)

#####  Method for realtime camera using Window OS

# cap = cv2.VideoCapture(0)
# cap.set(10 , 150)

# while True:
#     _, frame = cap.read()
#     cv2.imshow("RealTimeScan", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('w'):
#         get_dir = os.listdir(".\Images\Scanned")
#         numberofFiles = len(get_dir)
#         numberofFiles += 1
#         cv2.imwrite(".\Images\Scanned\output%d.jpg" %numberofFiles, frame)
#         Scanner = DocScanner(interactive=True)
#         Scanner.scan(".\Images\Scanned\output%d.jpg" %numberofFiles)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

###### Method for realtime picamera only using raspbian OS 
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
#         Scanner.scan("./Images/Scanned/output%d.jpg" %numberofFiles)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

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