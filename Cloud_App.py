import tensorflow as tf
import numpy as np
import cv2 as cv
import queue
from queue import Queue
import threading
from threading import Thread, Lock
from time import sleep
import socket
import requests
import base64
import json
import datetime
import pytz
import os
import sys
import random
import math
import coco
import utils
import model as modellib

#List of Class Names for Class IDs
class_names = ('BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
Image_Lock = Lock()
class Database:
	def __init__(self):
		self.ObjDict = {}
		
	def __Extract(self, Category, Model, Img):
		#########################################CONSIDER ADDING A LOCK HERE##############################################
		global Image_Lock
		with Image_Lock:
			Result = Model.detect([Img])
		
		#Test for bad image
		if(len(Result[0]['class_ids'])==0):
			raise
		
		#Determine Index of Object Category in List of Detected Class_IDs. 
		#If user specified object category was not detected in image use first detected class as default
		Detections = np.where(Result[0]['class_ids'] == class_names.index(Category))
		Index = Detections[0][0] if len(Detections[0] > 0) else 0
			
		Mask = Result[0]['masks'][:,:,Index]
		h,w = Mask.shape[0:2]
		New = np.zeros(w*h*3).reshape((h,w,3))
		for i in range(3):
		    New[:,:,i] = Mask
		    
		#Apply Mask to Img
		Img[New==0] = 0
		
		#Extract ORB Features
		Orb = cv.ORB_create()
		_, Desc = Orb.detectAndCompute(cv.cvtColor(Img, cv.COLOR_BGR2GRAY), None)
		
		#Compute Color Histogram
		Hist = cv.calcHist([Img], [0,1,2], Mask.astype(np.uint8), [16,16,16], [0,256,0,256,0,256])
		Hist = cv.normalize(Hist, None).flatten()
		print("Features Extracted")
		return Desc, Hist 

	def SetStatus(self, Category, Name, Value):
		if(Category not in class_names):
		    raise KeyError        
		if(Name not in self.ObjDict[Category]):
		    raise KeyError
		if(Value!=0 and Value!=1 and Value!=2):
		    raise ValueError
		    
		self.ObjDict[Category][Name][0] = Value
		    
	def SetObjDict(self, Category, Name, Model, Img, Thresh, CThresh, Status):
		if(Category not in class_names):
		    raise KeyError
		if(Status!=0 and Status!=1 and Status!=2):
		    raise ValueError
		
		#Raise ValueError if invalid Threshold Value     
		Thresh = int(float((Thresh)))
		CThresh = int(float((CThresh)))
		
		#Extract Features
		Desc, Hist = self.__Extract(Category, Model, Img)
		print("Uploading Object")
		if Category not in self.ObjDict:
			self.ObjDict[Category] = {Name : [Status, Thresh, Desc, CThresh, Hist]}#Thresh and Feature Values can be grouped into a Tuple
		else:
			self.ObjDict[Category][Name] = [Status, Thresh, Desc, CThresh, Hist]

	def GetObject(self, Category):
		return self.ObjDict[Category]

	def RemoveObjectDict(self, Category, Name):
		del self.ObjDict[Category][Name]

CamLock = Lock()
CamLock2 = Lock()
class Camera: 
	Tracker = None #{Cat: {Name : [Time, New Image] Queue}}
	Model = None    #Tensorflow Model
	Server = "54.245.167.107:5000"
	CameraID = 1

	def __init__(self, Name, Ratio):
		self.Name = Name #Name/Location
		self.Ratio = Ratio
		self.ID = str(Camera.CameraID)
		Camera.CameraID+=1

	def __Extract(self, Img, Mask):
		#Apply Mask to Img
		Img[Mask==0] = 0

		#Extract ORB Features
		Orb = cv.ORB_create()
		_, Desc = Orb.detectAndCompute(Img, None)
		return Desc

	def __Analyze(self, Img, Time, Overwrite, DataObj):
		#Analyze Image
		######################CONSIDER ADDING A LOCK HERE#########################################################
		#To Do List: Either figure out a way to apply Mutual Exclusion to the model or upgrade to more powerful EC2 Instance and allow each separate camera thread to
		#have control over their own Copy of the Model (Alot of memory for a copy (~1 GB for each Model Copy?))
		#print("Running Model")
		global Image_Lock
		with Image_Lock:
			Result = Camera.Model.detect([Img])
		
		#Extract List of Detected Categories
		IDs = Result[0]['class_ids']
		Masks = Result[0]['masks']
		BBs = Result[0]['rois']
		
		#print(IDs)
		#Compare Detected Categories with those in DataBase
		for i,ID in enumerate(IDs):
		    try:
		    	ObjDict = DataObj.GetObject(class_names[ID])
		    except KeyError:
		        continue

		    print("Analyzing Image")
		    #Extract ORB Features
		    Features = self.__Extract(cv.cvtColor(Img, cv.COLOR_BGR2GRAY), Masks[:,:,i])
		    Hist_New = cv.calcHist([Img], [0,1,2], Masks[:,:,i].astype(np.uint8), [16,16,16], [0,256,0,256,0,256])
		    Hist_New = cv.normalize(Hist_New, None).flatten()
	
		    #Compare with those in all corresponding DataObj Database Queues
		    for name in ObjDict:
		        print("Comparing with images in Object Database")
		        Status = ObjDict[name][0]
		        if(Status == 0):
		            continue
 
		        THRESHOLD = ObjDict[name][1]
		        Desc = ObjDict[name][2]
		        CTHRESHOLD = ObjDict[name][3]
		        Hist_Old = ObjDict[name][4]

		        BFM = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
		        matches = BFM.knnMatch(Features, Desc, k=2)

		        Good_Matches = 0
		        #Determine Good Matches
		        for m,n in matches:
		            if(m.distance < self.Ratio*n.distance):
		                Good_Matches+=1                   

		        #Compare Color Histograms
		        HD1 = cv.compareHist(Hist_New, Hist_Old, cv.HISTCMP_CHISQR)
		        HD2 = cv.compareHist(Hist_Old, Hist_New, cv.HISTCMP_CHISQR)
		        HDistance = max(HD1,HD2)
			
		        print("Good ORB Descriptor Matches:", Good_Matches)
		        print("Color HDistance:", HDistance)
		        #Apply Match Thresholds to determine if there is a match
		        if(THRESHOLD <= Good_Matches and HDistance <= CTHRESHOLD):
		            print("Match Found!!!")
			    #Add bounding box to Img
		            Box = BBs[i]
		            Img = cv.rectangle(Img, (Box[1],Box[0]), (Box[3],Box[2]), (0,255,0), 3)
		            
		            global CamLock2
		            with CamLock2:#May not be necessary because Queues are already thread safe
		                #Add Image along with Snapshot Time to appropriate Tracker Queue
		                try:
		                    if(Status == 1):#Object is Detectable
		                        Camera.Tracker[class_names[ID]][name].get()
		                        Camera.Tracker[class_names[ID]][name].put((Time, Img))
		                    else:#Object is Trackable.
		                        Camera.Tracker[class_names[ID]][name].put((Time, Img), block=False)
		                except KeyError:
		                    NewQ = Queue(maxsize=Overwrite) if (Overwrite!=None) else Queue()
		                    NewQ.put((Time, Img))
		                    if(class_names[ID] not in Camera.Tracker):
		                    	Camera.Tracker[class_names[ID]] = {name : NewQ}
		                    else:
		                    	Camera.Tracker[class_names[ID]][name] = NewQ
		                except queue.Full:
		                    Camera.Tracker[class_names[ID]][name].get()
		                    Camera.Tracker[class_names[ID]][name].put((Time, Img), block=False)                                                     

	def Snapshot(self, Overwrite, DataObj):
		#Fetch Image from the Server with HTTP Requests
		global CamLock
		with CamLock:
		    print("Fetching Image")
		    IP = "http://" + Camera.Server
		    requests.put(IP, data=self.ID)
		    Response = requests.get(IP)
		    Image = base64.b64decode(json.loads(Response.text)['py/b64'])
		    Img = cv.imdecode(np.frombuffer(Image, np.uint8), 1)
		
		#Img = cv.imread("Cell_Phone.png")	
		#print(Img.shape)
		#Receive Time
		T = datetime.datetime.now(pytz.utc)
		Time = str(T.astimezone(pytz.timezone('US/Central')))

		#Analyze Image
		self.__Analyze(Img, Time, Overwrite, DataObj)	

	def GetName(self):
		return self.Name #Return Name/Location Info

	def GetRatio(self):
		return self.Ratio

	def SetRatio(self, Ratio):
		self.Ratio = Ratio

	def GetTracker(Category, Name):
		return Camera.Tracker[Category][Name]

	def SetTracker(Category, Name=None):
		#Purge Tracker Queue Or Delete Entire Tracker Queue				
		if(Name == None):
		    Cat = Camera.Tracker[Category]
		    for key in Cat:
		        Cat[key].queue.clear()
		else:
		    del Camera.Tracker[Category][Name]

class User:
	def __init__(self, Client, Client_Addr, Max_Frames=10, Frequency=0.05, DEFAULT_BUFF_SIZE=1024):
		self.Client = Client
		self.Client_Addr = Client_Addr
		self.Overwrite = Max_Frames
		self.Snap_Freq = Frequency*60
		self.Buffer = DEFAULT_BUFF_SIZE
		self.ThreadLock = Lock()
		self.Pipe = Queue()
		self.Message = ""
		self.Command = 0
		self.Check = 0

	def __Worker(self, Name, Ratio, DB):
		Cam = Camera(Name, Ratio)
		while(self.Command!=-1):#Exit Command
			#print("Current Command:",self.Command)
			if(self.Command==0):
				with self.ThreadLock:
					self.Check+=1
				while(self.Command==0):
					pass
		
			elif(self.Command==1):
				Cam.Snapshot(self.Overwrite, DB)				
				while(self.Command == 1):
					#Take Periodic Snapshots
					sleep(self.Snap_Freq)
					Cam.Snapshot(self.Overwrite, DB)	
			
			elif(self.Command==2 and Name==self.Message):
				break
			else:
				pass

	def __Upload(self, Data, DB, Threads):
		Category = Data[1].lower()
		Name = Data[2]
		Thresh = Data[3]
		CThresh = Data[4]
		Status = Data[5]
					   
		#Receive Uploaded Image
		Response = requests.get("http://"+Camera.Server+"/User/")#Verify
		Image = base64.b64decode(json.loads(Response.text)['py/b64'])
		Img = cv.imdecode(np.frombuffer(Image, np.uint8), 1)			

		#Temporarily hold each thread in place
		self.Command = 0
		while(self.Check<Threads):
			pass				
		try:					
			#Add New Object Information to Database
			########Logic needs to be added here to determine what to do if no categories are detected in the user provided image##############
			DB.SetObjDict(Category, Name, Camera.Model, Img, Thresh, CThresh, int(Status))
	
			#Confirmation Message
			self.Client.send(bytes("Image successfully uploaded and features successfully extracted.", "utf8"))				
		except (ValueError, KeyError):
			#Send Error Message
			self.Client.send(bytes("Error, either invalid category, invalid threshold value(s), or invalid status.", "utf8"))
		except Exception:
			#Send Other Error Message
			self.Client.send(bytes("Error, something went wrong. Try to not upload a garbage image.", "utf8"))
		finally:
			#Release Threads
			self.Check = 0
			self.Command = 1	
	
	def __Download(self, Data, Threads):
		#Send Object Info Back to User
		Category = Data[1].lower()
		Name = Data[2]

		#Temporarily hold each thread in place
		self.Command = 0
		while(self.Check<Threads):
			pass				
		try:
			ObjQ = Camera.GetTracker(Category, Name)
			self.Client.send(bytes(str(ObjQ.qsize()), "utf8"))
			while not ObjQ.empty():
				Info = ObjQ.get()
				self.Client.recv(self.Buffer).decode("utf8")
				self.Client.send(bytes(str(Info[0]), "utf8"))
				_, Img_encoded = cv.imencode(".png", Info[1])
				IP = "http://"+Camera.Server+"/User/"
				requests.put(IP, data=Img_encoded.tostring())#Verify
		
			#Confirmation Message
			self.Client.send(bytes("Object information retrieved.", "utf8"))
		except KeyError:
			#Error Message
			self.Client.send(bytes("Error, object does not exist.", "utf8"))
		finally:
			#Release Threads
			self.Check = 0
			self.Command = 1
	
	def __Clear(self, Data, Threads):
		#Clear appropriate areas of Camera Tracker for Objects
		Category = Data[1].lower()
		Name = Data[2] if(Data[2]!="_") else None

		#Temporarily hold each thread in place
		self.Command = 0
		while(self.Check<Threads):
			pass
		try:
			#Delete Tracker Category Queues or Specific Queue with Category and Name
			Camera.SetTracker(Category, Name)
	
			#Confirmation Message
			self.Client.send(bytes("Deletion successful.", "utf8"))
		except KeyError:
			#Send Error Message
			self.Client.send(bytes("Error, invalid category or name.", "utf8"))
		finally:
			#Release Threads
			self.Check=0
			self.Command = 1
	
	def __Status(self, Data, DB, Threads):
		#Set Object Status
		Category = Data[1].lower()
		Name = Data[2]
		Status = Data[3]

		#Temporarily hold each thread in place
		self.Command = 0
		while(self.Check<Threads):
			pass				
		try:					
			#Clear Camera Tracker of Current Images Relating to Previous Status
			if(Category in Camera.Tracker):
				if(Name in Camera.Tracker[Category]):
					Camera.SetTracker(Category, Name)
			
			#Add New Object Information to Database
			DB.SetStatus(Category, Name, int(Status))
	
			#Confirmation Message
			self.Client.send(bytes("Object Status successfully updated.", "utf8"))				
		except (ValueError, KeyError):
			#Send Error Message
			self.Client.send(bytes("Error, either category is invalid, ojbect name doesn't exist within category subset, \
			or invalid status.", "utf8"))
		finally:
			#Release Threads
			self.Check = 0
			self.Command = 1
			
	def Run(self, DB):
		Img = cv.imread("Camera1.png")
		Camera.Model.detect([Img])
		print("Server is ready")
		Threads = {}
		#The Interface with the Client and the Control Center of the Entire Program/Cloud Application		
		while(True):
			Data = self.Client.recv(self.Buffer).decode("utf8")
			#print("Message From User:", Data)
			if(Data==''):
				#Connection Lost Shutdown Server
				self.Command=-1
				print("Error, connection to Client lost. Shutting down now.")
				for thread in Threads:
					Threads[thread].join()
			
				break

			Data = Data.split(",")
			if(Data[0].lower()=="add"):
				#Create new thread and throw it into Worker Function
				Name = Data[1]
				Ratio = float(Data[2])
		
				#Create New Camera Thread
				self.Command = 1
				Worker = Thread(target=self.__Worker, args=(Name, Ratio, DB))
				Threads[Name] = Worker
				Worker.start()
		
				#Confirmation Message
				self.Client.send(bytes("Camera successfully added.", "utf8"))	
				
			elif(Data[0].lower()=="upload"):
				self.__Upload(Data, DB, len(Threads))
										
			elif(Data[0].lower()=="info"):
				self.__Download(Data, len(Threads))					
							
			elif(Data[0].lower()=="clear"):
				self.__Clear(Data, len(Threads))
				
			elif(Data[0].lower()=="status"):
				self.__Status(Data, DB, len(Threads))
									
			elif(Data[0].lower()=="delete"):
				#Delete Camera
				Name = Data[1]
				
				if(Name not in Threads):
					self.Client.send(bytes("Error, Camera does not exist.", "utf8"))
				else:
					#Temporarily hold each thread in place
					self.Message = Name
					self.Command = 2
					Threads[Name].join()
					del Threads[Name]#Make sure exception isn't thrown here
					self.Message = ""
					self.Command = 1
			
					#Confirmation Message
					self.Client.send(bytes("Camera successfully removed from network.", "utf8"))
	
			elif(Data[0].lower()=="shutdown"):
					#Initiate Server Shutdown
					self.Command=-1
					for thread in Threads:
						Threads[thread].join()
			
					#Confirmation Message
					self.Client.send(bytes("Cameras successfully removed from network. System shutting down.", "utf8"))					
					break
			
			else:
				self.Client.send(bytes("Parsing Error.", "utf8"))						

class InferenceConfig(coco.CocoConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

#Dictionary to initialize Camera Tracker
Track = {'BG' : {}, 'person' : {}, 'bicycle': {}, 'car': {}, 'motorcycle': {}, 'airplane': {}, 'bus': {}, 
		 'train': {}, 'truck': {}, 'boat': {}, 'traffic light': {}, 'fire hydrant': {}, 'stop sign': {}, 
		 'parking meter': {}, 'bench': {}, 'bird': {}, 'cat': {}, 'dog': {}, 'horse': {}, 'sheep': {},
		 'cow': {}, 'elephant': {}, 'bear': {}, 'zebra': {}, 'giraffe': {}, 'backpack': {}, 'umbrella': {},
		 'handbag': {}, 'tie': {}, 'suitcase': {}, 'frisbee': {}, 'skis': {}, 'snowboard': {}, 'sports ball': {},
		 'kite': {}, 'baseball bat': {}, 'baseball glove': {}, 'skateboard': {}, 'surfboard': {}, 'tennis racket': {},
		 'bottle': {}, 'wine glass': {}, 'cup': {}, 'fork': {}, 'knife': {}, 'spoon': {}, 'bowl': {}, 'banana': {},
		 'apple': {}, 'sandwich': {}, 'orange': {}, 'broccoli': {}, 'carrot': {}, 'hot dog': {}, 'pizza': {},
		 'donut': {}, 'cake': {}, 'chair': {}, 'couch': {}, 'potted plant': {}, 'bed': {}, 'dining table': {},
		 'toilet': {}, 'tv': {}, 'laptop': {}, 'mouse': {}, 'remote': {}, 'keyboard': {}, 'cell phone': {},
		 'microwave': {}, 'oven': {}, 'toaster': {}, 'sink': {}, 'refrigerator': {}, 'book': {}, 'clock': {},
		 'vase': {}, 'scissors': {}, 'teddy bear': {}, 'hair drier': {}, 'toothbrush': {}}
	
if __name__ == '__main__':
	DB = Database()
	ROOT_DIR = os.getcwd()
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)

	config = InferenceConfig()
	Camera.Model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
	Camera.Model.load_weights(COCO_MODEL_PATH, by_name=True)
	Camera.Tracker = Track	

	print("Successfully loaded model. Connecting to user now")
	Server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	Server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	Server.bind(('0.0.0.0', 10000)) 

	#This part of the program can be changed in the future to accept connections from multiple users and divide each user load into
	#a separate process
	Server.listen(1)
	Client, Client_Addr = Server.accept()
	print("Connection Established") 
	User1 = User(Client, Client_Addr)
	User1.Run(DB)

	Server.shutdown(0)         
	Server.close() 
	

'''				 
#Options
MAX_FRAMES = 60
FREQUENCY = 0.5 #By Default Snap Once every 30 seconds (Min/Snap)
'''

'''
Functions to Implement:
Workers:
	Take Periodic Snapshots
Dispatcher:	
	Essential:
		Add Camera
		Upload Image and Add to Object Database
		Get Object Info
	Non-Essential But What I Would Still Like To Iplement:
		Clear Tracker Category or Delete Object Queues
		Set Object Status (Object Tracker History is cleared here)		
	Non-Essential To Be Added If Time Allows:	
		Change Frequency of Camera Snaps
		Change Maximum number of Pictures in Camera Histories (Entire Tracker Dictionary will be cleared here)
		Remove from Object Database
		Get Descriptor Ratio
		Set Descriptor Ratio	
'''


