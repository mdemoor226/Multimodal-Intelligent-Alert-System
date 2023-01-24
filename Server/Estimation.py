import numpy as np
import cv2 
import sys
import os
from sys import platform
sys.path.append('../../python');
from openpose import pyopenpose as op

#Refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md on what joint each KeyPoint represents
#The methods above will return Keypoints for every detected person in the image. Each Keypoint will contain an xy coordinate along
#with a Confidence Level prediction. About 25 Keypoints per person (at least for the body pose excluding the hand/face Keypoints).

class Estimator(object):
    def __init__(self, model_path="./../../../models", face=True, hands=True):
        self.face = face
        self.hands = hands

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict()
        self.params["model_folder"] = model_path #"../../../model/"
        self.params["face"] = face
        self.params["hand"] = hands
        
        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()
        
        # Initialize Estimation Datastructure
        self.datum = None
    
    def Analyze(self, image):
        self.datum = op.Datum()
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop([self.datum])
        
    def GetBody(self):
        assert self.datum is not None, "Error, analyze an image first before making calls."
        return self.datum.poseKeypoints
    
    def GetHands(self):
        assert self.datum is not None, "Error, analyze an image first before making calls."
        assert self.hands, "Hand Keypoints set to False."
        return self.datum.handKeypoints # (LeftHand, RightHand)
    
    def GetFace(self):
        assert self.datum is not None, "Error, analyze an image first before making calls."
        assert self.face, "Face Keypoints set to False."
        return self.datum.faceKeypoints
    
    def Visualize(self):
        assert self.datum is not None, "Error, analyze an image first before making calls."
        return self.datum.cvOutputData


class FaceRemover(object):
    def __init__(self, face_model_path="./../../../../../res10_300x300_ssd_iter_140000.caffemodel", face_proto_path="./../../../../../deploy.prototxt.txt",
                 pose_model_path="./../../../models", hands=False):
        self.hands = hands
        
        #Human Pose Estimation
        print("Initializing OpenPose...")
        self.HEstimator = Estimator(model_path=pose_model_path, hands=hands)
        
        #Face Detection
        print("Loading and Initializing Face Detector...")
        self.FDetector = cv2.dnn.readNetFromCaffe(face_proto_path, face_model_path)
        print("Face Detector Loaded.")
        
    def getFacePoints(self, Image):
        self.HEstimator.Analyze(Image)
        return self.HEstimator.GetFace()

    def GetKeyPoints(self, Image):
        self.HEstimator.Analyze(Image)
        Hands = self.HEstimator.GetHands() if self.Hands else None
        Face = self.HEstimator.GetFace()
        Body = self.HEstimator.GetBody()
        
        return Body, Face, Hands
    
    def BlurFaces(self, Image, FacePoints=None):
        if FacePoints is None:
            FacePoints = self.getFacePoints(Image)
        
        if len(FacePoints.shape) == 0:
            #There were no Faces Detected by the Pose Estimation Algorithm
            return Image, None
        
        #Create Polygon Coordinates from Face Keypoints to Segment out Facial Areas - # 0-16, 26-17, (17,0)
        #And Obtain Boxes for Facial Blurring
        Img = np.copy(Image)
        h, w = Image.shape[:2]
        SegMasks = np.empty((h,w,3,FacePoints.shape[0]))
        for i in np.arange(FacePoints.shape[0]):
            Segment = np.zeros_like(Image)
            Points = FacePoints[i,:,:2]
            Polys = np.concatenate((Points[:17,:],Points[17:27,:][::-1]), axis=0).astype(np.int32)
            
            #Clamp to within image boundaries
            Polys[Polys < 0] = 0
            Polys[:,0][Polys[:,0] > w] = w - 1
            Polys[:,1][Polys[:,1] > h] = h - 1

            #Obtain Box Coordinates for Cropping and Blurring
            Minx, Miny = np.amin(Polys, axis=0)
            Maxx, Maxy = np.amax(Polys, axis=0)
            Img[Miny:Maxy+1,Minx:Maxx+1,:] = cv2.GaussianBlur(Image[Miny:Maxy+1,Minx:Maxx+1,:], (63,63), 30)

            #Create Segmentation Masks from Polygons
            cv2.fillPoly(Segment, pts=[Polys], color=(1,1,1))
            SegMasks[:,:,:,i] = Segment

        #Only apply the Blurred Pixels belonging to Segmented Faces onto the Original Image
        Union = np.amax(SegMasks, axis=3)
        Image[Union>0] = Img[Union>0]

        return Image, SegMasks
    
    def DetectAndBlur(self, Image, CTHRESHOLD=0.3, IoUThresh=0.4):
        ############ Inputs #############
        ## Image : Image to Analyze #####
        ## CTHRESHOLD : Acceptable ######
        ## Confidence Level for Face ####
        ## Predictions ##################
        #################################
        
        FLAG = False #Flag the Frame for User Review
        FacePoints = self.getFacePoints(Image) #Predict Face KeyPoints
        KeyFaces = 0 if len(FacePoints.shape)==0 else FacePoints.shape[0]
        
        #Use Facial Keypoints to Segment and Blur out Faces in the Image
        BlurredImg, SegMasks = self.BlurFaces(np.copy(Image), FacePoints)
        
        #Normalize Input Image/Data Before Forwarding Through the Separate Face Detection Network
        h, w = Image.shape[:2]
        Data = cv2.dnn.blobFromImage(cv2.resize(Image, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        self.FDetector.setInput(Data)
        Detections = self.FDetector.forward()
        
        #Process Face Detections - Some Code here (a few lines) is adapted off of Adrian Rosebrocks PyImageSearch Tutorials
        Prediction_Count = 0
        FaceMaps = np.array([False for _ in range(KeyFaces)])
        for i in np.arange(Detections.shape[2]):
            #Use the Predicted Probability as a Confidence Threshold for Suppressing Weak Predictions
            Confidence = Detections[0, 0, i, 2]
            if Confidence >= CTHRESHOLD:
                Prediction_Count+=1
                
                #Obtain Bounding Box Coordinates for Predicted Face
                BBox = Detections[0, 0, i, 3:7] * np.array([w, h, w, h]) #BBox = (StartX, StartY, EndX, EndY)
                BBox[BBox < 0] = 0
                BBox[2] = w - 1 if BBox[2] > w else BBox[2]
                BBox[3] = h -1 if BBox[3] > h else BBox[3]
                BBox = BBox.astype("int")
                
                #If the IoU overlap between all the predicted Face Masks and the given Bounding Box is less than some Threshold, Blur the whole Box and Flag the Frame
                if KeyFaces > 0:
                    #Calculate the IoUs between the Predicted Bounding Box and each of the Predicted Face Segmentation Masks
                    UBox = np.zeros((h,w,1), dtype=SegMasks.dtype)
                    UBox[BBox[1]:BBox[3],BBox[0]:BBox[2]] = 1
                    Unions = np.logical_or(SegMasks[:,:,0,:], UBox)
                    IoUs = np.sum(SegMasks[BBox[1]:BBox[3],BBox[0]:BBox[2],0,:], axis=(0,1)) / np.sum(Unions, axis=(0,1)) #((BBox[2] - BBox[0])*(BBox[3] - BBox[1]))
                    if np.all(np.less(IoUs, IoUThresh)):
                        BlurredImg[BBox[1]:BBox[3],BBox[0]:BBox[2],:] = cv2.GaussianBlur(Image[BBox[1]:BBox[3],BBox[0]:BBox[2],:], (63,63), 30)
                        FLAG = True

                    #Check which person/people that the KeyPoints that exist belong to
                    Points = FacePoints[:,:,:2]
                    MinPoints = np.greater(Points - np.array([[[BBox[0], BBox[1]]]], dtype=Points.dtype),0)
                    MaxPoints = np.less(Points - np.array([[[BBox[2], BBox[3]]]], dtype=Points.dtype),0)
                    TruthMap = np.logical_and(MinPoints[:,:,0], MinPoints[:,:,1]) & np.logical_and(MaxPoints[:,:,0], MaxPoints[:,:,1])
                    for i in np.arange(KeyFaces):
                        if np.any(TruthMap[i,:]):
                            FaceMaps[i] = True
                        
                else:
                    #No Face Keypoints exist at all. So just Blur the whole box.
                    BlurredImg[BBox[1]:BBox[3],BBox[0]:BBox[2],:] = cv2.GaussianBlur(Image[BBox[1]:BBox[3],BBox[0]:BBox[2],:], (63,63), 30)
                                        
        #If all the Face Keypoints for a particular person fall within no predicted Box Flag the Frame
        if not np.all(FaceMaps):
            FLAG = True
        
        #If either no Facial Keypoints are predicted or no Faces are Predicted Flag the Frame
        if KeyFaces == 0 or Prediction_Count == 0:
            FLAG = True
        
        ###################### Returns ###########################
        ####### Image : The Image with Blurred Faces #############
        ####### FLAG : Flag the frame to suggest user review #####
        ##########################################################
        #self.HEstimator.Analyze(Image)
        #return self.HEstimator.Visualize(), False
        return BlurredImg, FLAG           

if __name__=='__main__':
    Test = cv2.imread("./Example2.jpg")
    Privacy = FaceRemover()
    Image, Flag = Privacy.DetectAndBlur(Test)
    cv2.imshow("Facial Blurring Example", Image)
    cv2.waitKey(0)
    
    
