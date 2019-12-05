import socket
import numpy as np
import cv2 as cv
from time import sleep
from threading import Thread, Lock
import base64
#import matplotlib.pyplot as plt
import imagezmq

#List of Categories #Possible Demo Choices: Cell Phone, Laptop, Mouse, Keyboard, Book
class_names = ["background", "airplane", "bicycle", "bird", "boat",
	   "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
	   "dog", "horse", "motorcycle", "person", "potted plant", "sheep",
	   "sofa", "train", "tv"]

CLASS_NAMES = ('BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush')

FRAME = None
STREAM = True
def CollectVideo():
    global FRAME

    #Initialize ImageHub
    print("Awaiting Incoming Connection...")
    ImageHub = imagezmq.ImageHub(open_port='tcp://*:5570')

    #Stream Loop
    while STREAM:
        #Recieve Server Name and Processed Frame and acknowledge with a reply
        (CamName, Frame) = ImageHub.recv_image()
        ImageHub.send_reply(b'OK')
        FRAME = Frame
    
    return

def Display():
    while True:
        if FRAME is None:
            continue
        
        cv.imshow("Camera View(s)", FRAME)
        Key = cv.waitKey(1) & 0xFF
        if Key == ord("q"):
            break
    
    cv.destroyAllWindows()

def Verify_Input():
    while(True):
        try:
            Value = float(input())
            return str(Value)
        except ValueError:
            print("Error, invalid entry. Must be floating point number.")#(ideally between 0.5-0.9)

##################################################################################################################################################################
UPLOADFRAME = np.zeros((640,480), dtype=np.uint8)
def UploadServer():
    # Send Image to the Backend Cloud App
    SERVER_IP = "10.3.12.31"#"172.24.118.97" #Change this as necessary

    #Initialize Sender Object for the Server
    print("Connecting to User...")
    sleep(10)
    Sender = imagezmq.ImageSender(connect_to="tcp://{}:5556".format(SERVER_IP))

    #Obtain Hostname, initialize Video Stream, and Warm Up the Camera
    ServerName = socket.gethostname()
    
    #Send the Processed Image Frames Back to the Client
    while STREAM:       
        Sender.send_image(ServerName, UPLOADFRAME)
    
    return

def Upload(Conn, Send):
    global UPLOADFRAME

    #Upload Image to WebServer
    print("Enter the path to the image of the object you would like to upload.")
    while(True):
        Filename = input()
        Img = cv.imread(Filename)
        if Img is not None:
            UPLOADFRAME = Img	
            break
                
        print("Error, file does not exist.")
	
    print("Short Class List or Long?")
    while(True):
        Answer = input().lower()
        if Answer in {"short", "long"}:
            break

        print("Please enter Short or Long?")

    Classes = class_names if Answer=="short" else CLASS_NAMES
        
    print("Enter a category")
    while(True):
        Category = input()
        if(Category in Classes):
            break
        print("Error, invalid category. Select a category from the following.")
        print(Classes)
    
    Name = input("Enter a name: ")
	
    Thresh = "10"
    print("Would you like to fine-tune the accuracy of object recognition by specifying a threshold for good ORB descriptor matches? Default = 10 (The higher the more accurate but less chance of detection.")
    while(True):
        Answer = input()
        if(Answer.lower()=="yes"):
            print("Enter new threshold value.")
            Thresh = Verify_Input()
            break
        elif(Answer.lower()=="no"):
            break
        else:
            print("Error, please enter yes or no.")
	
    CThresh = "25"
    print("Would you like to further fine-tune the accuracy of object recognition by specifying a threshold for color histogram matching? Default = 25 (The lower the more accurate but less chance of detection.")
    while(True):
        Answer = input()
        if(Answer.lower()=="yes"):
            print("Enter new threshold value.")
            CThresh = Verify_Input()
            break
        elif(Answer.lower()=="no"):
            break
        else:
            print("Error, please enter yes or no.")
	
    Status = "Nothing"
    print("Specify the tracking status of the object. Should the application track the object over time (Track), simply detect it and relay its most recently known location (Detect), or do nothing at all with respect to the object (Nothing).")
    while(True):
        Answer = input()
        if(Answer.lower()=="track"):
            Status = "2"
            break
        elif(Answer.lower()=="detect"):
            Status = "1"
            break
        elif(Answer.lower()=="nothing"):
            Status = "0"
            break
        else:
            print("Error, please enter \"Detect\",\"Track\", or \"Nothing\".")
    
    Send = Send+","+Category+","+Name+","+Thresh+","+CThresh+","+Status
    Conn.send(bytes(Send, "utf8"))
    
    Response = Conn.recv(1024).decode("utf8")
    print(Response)	
##################################################################################################################################################################

##################################################################################################################################################################
NUMFRAMES = 0
DLock = Lock()
DOWNLOADFRAMES = []
def DownloadServer():
	global DOWNLOADFRAMES, NUMFRAMES, DLock

	#Initialize ImageHub
	print("Awaiting Incoming Connection...")
	ImageHub = imagezmq.ImageHub(open_port='tcp://*:5560')

	#Stream Loop
	while STREAM:
		with DLock:
			Count = 0
			while Count < NUMFRAMES:		
				#Recieve camera name and Downloaded Image Frame and acknowledge with a receipt reply
				(CamName, Download) = ImageHub.recv_image()
				ImageHub.send_reply(b'OK')
				#if Download is Nonsense:
				#	continue
				
				DOWNLOADFRAMES.append(Download)
				Count+=1
		
			NUMFRAMES = 0
	
	return

def Download(Conn, Send):
    global DOWNLOADFRAMES, NUMFRAMES, DLock

    print("Short Class List or Long?")
    while(True):
        Answer = input().lower()
        if Answer in {"short", "long"}:
            break
        
        print("Please enter Short or Long?")

    Classes = class_names if Answer=="short" else CLASS_NAMES
        
    #Send Category and Name
    print("Enter a category")
    while(True):
        Category = input()
        if(Category in Classes):
            break
        print("Error, invalid category. Select a category from the following.")
        print(Classes)
    
    Name = input("Enter a name: ")
    
    Send = Send+","+Category+","+Name
    Conn.send(bytes(Send, "utf8"))
    
    #Try to get number of images to retrieve from Server
    Response = Conn.recv(1024).decode("utf8")
    try:
        Num = int(float(Response))

        Times = []
        for _ in range(Num):
            #Receive Time Stamp
            Conn.send(bytes("Handshake", "utf8"))
            Times.append(Conn.recv(1024).decode("utf8"))
            sleep(0.25)
        
        print("Retrieved Times!!!")
        #Retrieve Object Information and Images
        with DLock:
            DOWNLOADFRAMES.clear()
            NUMFRAMES = Num
        
        sleep(10)
        Images = DOWNLOADFRAMES.copy()
        Response = Conn.recv(1024).decode("utf8")		
        print(Response)

        #This part can be modified in the future to determine how to properly display information to the user.
        for i,Result in enumerate(Images):
            cv.imshow("Camera: "+Name+" "+Times[i], Result)
            #plt.title("Camera: "+Name+" "+Times[i])
            #plt.show()
            cv.waitKey(0)
                    
    except ValueError:
        print(Response)
##################################################################################################################################################################

def Add(Conn, Send):
	Name = input("Enter the name of the new Camera: ")
	Ratio = "0.75"
	print("Would you like to change the accuracy ratio of the descriptor matching algorithm? Default = 0.75")
	while(True):
		Answer = input()
		if(Answer.lower()=="yes"):
			print("Enter new accuracy ratio.")
			Ratio = Verify_Input()
			break
		elif(Answer.lower()=="no"):
			break
		else:
			print("Error, please enter yes or no.")
	
	Send = Send+","+Name+","+Ratio
	Conn.send(bytes(Send, "utf8"))
	
	Response = Conn.recv(1024).decode("utf8")
	print(Response)

def Speed(Conn, Send):
    print("Set a particular Camera into FAST/SLOW mode?")
    while True:
        Answer = input()
        if Answer.lower() == "no":
            return
        elif Answer.lower() == "yes":
            break
        else:
            print("Please enter yes or no.")
    
    Name = input("Which Camera? ")
    
    print("FAST or SLOW?")
    while True:
        Answer = input().lower()
        if Answer in {"fast", "slow"}:
            break
        
        print("Please enter FAST or SLOW.")
    
    Send = Send+","+Name+","+Answer
    Conn.send(bytes(Send, "utf8"))
    
    #Await Response
    Response = Conn.recv(1024).decode("utf8")
    print(Response)
        
def Clear(Conn, Send):
    print("Short Class List or Long?")
    while(True):
        Answer = input().lower()
        if Answer in {"short", "long"}:
            break
        
        print("Please enter Short or Long?")
    
    Classes = class_names if Answer=="short" else CLASS_NAMES
    
    print("Enter a category")
    while(True):
        Category = input()
        if(Category in Classes):
            break
        print("Error, invalid category. Select a category from the following.")
        print(Classes)	
    
    print("Would you like to specify an object name and delete that object's history? No will result in the histories of all objects belonging to the category specified above to be deleted.")
    while(True):
        Answer = input()
        if(Answer.lower()=="yes"):
            print("\nEnter an object name")
            Name = input()
            break
        elif(Answer.lower()=="no"):
            Name = "_"
            break
        else:
            print("Error, please enter yes or no")
    
    Send = Send+","+Category+","+Name
    Conn.send(bytes(Send, "utf8"))
    Response = Conn.recv(1024).decode("utf8")
    print(Response)
	
def Status(Conn, Send):
    print("Short Class List or Long?")
    while(True):
        Answer = input().lower()
        if Answer in {"short", "long"}:
            break
        
        print("Please enter Short or Long?")
    
    Classes = class_names if Answer=="short" else CLASS_NAMES
    
    print("Enter a category")
    while(True):
        Category = input()
        if(Category in Classes):
            break
        print("Error, invalid category. Select a category from the following.")
        print(Classes)
    
    Name = input("Enter a name: ")	
    
    Status = "Nothing"
    print("Specify the tracking status of the object. Should the application track the object over time (Track)?, simply detect it and relay its most recently known location (Detect)?, or do nothing at all with respect to the object (Nothing)?.")
    while(True):
        Answer = input()
        if(Answer.lower()=="track"):
            Status = "2"
            break
        elif(Answer.lower()=="detect"):
            Status = "1"
            break
        elif(Answer.lower()=="nothing"):
            Status = "0"
            break
        else:
            print("Error, please enter \"Detect\",\"Track\", or \"Nothing\".")
    
    Send = Send+","+Category+","+Name+","+Status
    Conn.send(bytes(Send, "utf8"))
    Response = Conn.recv(1024).decode("utf8")
    print(Response)
	
def Delete(Conn, Send):
    Name = input("Enter Camera Name: ")
    
    Send = Send+","+Name		
    Conn.send(bytes(Send, "utf8"))
    Response = Conn.recv(1024).decode("utf8")
    print(Response)

def Shutdown(Conn, Send):
    Conn.send(bytes(Send, "utf8"))
    Response = Conn.recv(1024).decode("utf8")
    print(Response)
	
if __name__ == '__main__':    
    print("Connecting to Server")
    Client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #Client.connect(("127.0.0.1",10000)) #Workstation User App
    Client.connect(("10.3.12.31",10000)) #Other User App
    print("Connection Established")

    #Start collecting Video Frames to Display upon the User's Request
    Collector = Thread(target=CollectVideo)
    Uploader = Thread(target=UploadServer)
    Downloader = Thread(target=DownloadServer)
    Collector.start()
    Uploader.start()
    Downloader.start()

    #Start the User Application
    print("Welcome to the IoT Object Tracker. Type help for a list of commands.")
    while(True):
            In = input()
            if(In.lower()=="add"):
                    Add(Client, In)
            elif(In.lower()=="speed"):
                    Speed(Client, In)
            elif(In.lower()=="upload"):
                    Upload(Client, In)
            elif(In.lower()=="download"):
                    Download(Client, "Info")
            elif(In.lower()=="clear"):
                    Clear(Client, In)
            elif(In.lower()=="status"):
                    Status(Client, In)
            elif(In.lower()=="delete"):
                    Delete(Client, In)
            elif(In.lower()=="display"):
                    Display()
            elif(In.lower()=="shutdown"):
                    STREAM = False
                    Shutdown(Client, In)
                    break
            elif(In.lower()=="help"):
                    print("Add: Add a new camera to the network\nUpload: Upload a personal picture of an object\nDownload: Download information about an objectClear: Clear histories of a particular object or object category.\nStatus: Change the track status of an object\nDelete: Remove a camera from the network\nShutdown: Shutdown application\nHelp: Display a list of commands\nView: View category/class names")
            elif(In.lower()=="view"):
                    print("Short List:\n", class_names)
                    print("Long List:\n", CLASS_NAMES)
            else:
                    print("Error, invalid entry. Type help for a list of commands.")
            
            print()
	
    #Close Connection to Backend and Collect Threads
    Client.shutdown(0)
    Client.close()
    Collector.join()
    Uploader.join()
    Downloader.join()



