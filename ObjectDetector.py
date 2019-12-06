import numpy as np
import cv2
import imagezmq
from datetime import datetime
from Estimation import Estimator
import socket
from time import sleep
from threading import Thread
#import caffe
#caffe.set_mode_gpu()
#caffe.set_device(0)

#Much of the Object Detection and Message Passing Code here is adapted from Adrian Rosebrocks PyImageSearch Tuturials and ImUtils Library:
#https://github.com/jrosebr1/imutils

# List of Class Labels that MobileNet SSD was trained to detect
CLASSES = ["background", "airplane", "bicycle", "bird", "boat",
	   "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
	   "dog", "horse", "motorcycle", "person", "potted plant", "sheep",
	   "sofa", "train", "tv"]

STREAM = False
FRAME = None
def Montagizer(Images, IShape, MShape):
	h, w = IShape
	mh, mw = MShape
	
	#Squeeze as many images as possible into Montage frame. Once no more fit create a new Montage frame and start again.
	ypos, xpos = 0, 0
	ymax, xmax = h*mh, w*mw
	Montage = np.zeros((mh*h, mw*w, 3), dtype=np.uint8)
	for Img in Images:
		#Position resized images in montage frame(s) in cycles starting from top left (0,0) going down and right.
		Img = cv2.resize(Img, (w,h))
		Montage[ypos:ypos+h,xpos:xpos+w,:] = Img
		xpos += w
		if xpos >= xmax:
			ypos += h
			xpos = 0
	
	return Montage

def Server(NumCams=1, CThreshold=0.7):
    global STREAM, FRAME
    assert 0 < NumCams <= 4, "Error, only a maximum of 4 (and a minimum of 1) cameras are supported at this time."
    if NumCams == 1:
        MWidth = 1
        MHeight = 1
    elif NumCams == 2:
        MWidth = 1
        MHeight = 1
    else:
        MWidth = 2
        MHeight =2

    print("Standby, loading Object Detector...")
    NN = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt", "./MobileNetSSD_deploy.caffemodel")

    #Initialize ImageHub
    print("Awaiting Incoming Connection...")
    ImageHub = imagezmq.ImageHub(open_port='tcp://*:5580')

    print("Standby, loading Object Detector...")
    NN = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt", "./MobileNetSSD_deploy.caffemodel")

    #Frame Dictionary for storing the different frames from different cameras
    FrameMap = {}
    
    #Initialize Human Pose Estimator
    HEstimator = Estimator()
    
    #Stream Loop
    STREAM = True
    while STREAM:
        #Recieve camera name and acknowledge with a receipt reply
        (CamName, Frame) = ImageHub.recv_image()
        ImageHub.send_reply(b'OK')
        
        #print("Image Received!!!")
        #print(Frame)
        #Check if new data is coming from a newly connected device
        if CamName not in FrameMap:
            print("Recieving new data from {}...".format(CamName))
        
        #Resize the image frame to have a width of 400 pixels and then normalize the data before forwarding through the Neural Network
        h, w = Frame.shape[:2]			
        Ratio = 400.0 / float(w)
        #Frame = cv2.resize(Frame, (400, int(h*Ratio)), cv2.INTER_AREA)
        Data = cv2.dnn.blobFromImage(cv2.resize(Frame, (300, 300)), 0.007843, (300, 300), 127.5)

        #Pass the Data through the MobileNet SSD Object Detector and obtain Predictions
        NN.setInput(Data)
        Detections = NN.forward()
        
        for i in np.arange(Detections.shape[2]):
            #Extract the Confidence Level (i.e. Probability) of the prediction
            Confidence = Detections[0, 0, i, 2]
                
            #Supress Weak Predictions (those with a Confidence less than a specified threshold)
            if Confidence >= CThreshold:
                #print("Detected Something!!!")
                
                #Extract Class Index
                index = int(Detections[0, 0, i, 1])
                Class = CLASSES[index]
                if Class == "person":
                    HEstimator.Analyze(Frame)
                    Frame = HEstimator.Visualize()
                
                #Extract BoundingBox Information
                BBox = Detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = BBox.astype('int')
                
                #Draw the Box on the Image Frame
                cv2.rectangle(Frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                #Label the Object on the Bounding Box
                cv2.putText(Frame, Class, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                #Save Test to File
                #cv2.imwrite("./Test.jpg", Frame)
        
        #Write the Device name to be displayed on the recieved Image Frame
        cv2.putText(Frame, CamName, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #Update most recent frame in Frame Dictionary
        FrameMap[CamName] = Frame
        
        #Construct the Montage from the Frames of every Active Camera
        h, w = Frame.shape[:2]
        Montage = Montagizer(FrameMap.values(), Frame.shape[:2], (MHeight, MWidth))
        
        #Send the Most Recent Processed Group of Frames Back to the User
        FRAME = Montage
        
        #Testing
        #cv2.imshow("Testing123...", Montage)
        #cv2.waitKey(1)

    #Nuke any leftover Open Windows in the Program
    #cv2.destroyAllWindows()
    
    return

def MessagePassing():
    global STREAM
    
    # Send Data Back to Client
    #SERVER_IP = "127.0.0.1" #Workstation User App
    SERVER_IP = "172.24.98.16" #Rpi User App #"172.24.118.97"

    #Initialize Sender Object for the Server
    print("Connecting to Client...")
    sleep(10)
    Sender = imagezmq.ImageSender(connect_to="tcp://{}:5570".format(SERVER_IP))

    #Obtain Hostname, initialize Video Stream, and Warm Up the Camera
    ServerName = socket.gethostname()
    
    #Send the Processed Image Frames Back to the Client
    while True:
        try:
            if FRAME is None:
                continue
            
            Frame = FRAME
            Sender.send_image(ServerName, Frame)

        except KeyboardInterrupt:
            print("Shutting down...")
            STREAM = False
            break

if __name__=='__main__':
    WebServer = Thread(target=Server)
    WebServer.start()
    MessagePassing()
    WebServer.join()
    #Server()
