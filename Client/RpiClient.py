import imagezmq
import socket
from time import sleep
from VideoStream import Stream
#import cv2

SERVER_IP = "10.3.12.31" #"12.92.11.78" #Change this

#Initialize Sender Object for the Server
print("Connecting to Server...")
Sender = imagezmq.ImageSender(connect_to="tcp://{}:5580".format(SERVER_IP))
Sender2 = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(SERVER_IP))
print("Connection Established.")

#Obtain Hostname, initialize Video Stream, and Warm Up the Camera
CamName = socket.gethostname()
vStream = Stream(Rpi=True).Start()
sleep(3)

#Start the Video Stream
while True:
	#Obtain Video Frame and send it to the Server (Backend)
        Frame = vStream.ReadFrame()
        Sender.send_image(CamName, Frame)
        Sender2.send_image(CamName, Frame)
        #cv2.imshow("Test", Frame)
        #cv2.waitKey(1)

