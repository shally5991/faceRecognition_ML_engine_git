from deepface.detectors import FaceDetector
from deepface import DeepFace
import cv2,os,time
import requests

#need to update the code for the database part once done with the files of the s3  buket uploaded via the register portal

class FaceRecognition:
    def __init__(self,camera,db_path,model_name,metric):
        self.detector_name = "opencv"
        self.detector = FaceDetector.build_model(self.detector_name)  # set opencv, ssd, dlib, mtcnn or retinaface
        self.cam_id=camera
        self.database=db_path
        self.model_name=model_name
        self.metric=metric
        self.post_url='http://localhost:1234/api/person/check_in'
        self.put_url='http://localhost:1234/api/person/check_out/'

    def recognition(self,image,enf_dect):        
        df=DeepFace.find(img_path=image,db_path=self.database,enforce_detection=enf_dect,model_name=self.model_name,distance_metric=self.metric)
        if len(df)>0:         
            
            print(df)
            name =  df[0].loc[0, 'identity'].split('/')[-1][:-4].rstrip('1')

            return name
        else:
            return "unkown"

    def in_streamer(self):
        cam = cv2.VideoCapture(self.cam_id)
        pt=time.time()
        fs=0
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontcolor = (0, 255, 0)
        new_name=None
        while True:
            res,img = cam.read()
            if not res:
                break
            fs+=1
            faces = FaceDetector.detect_faces(self.detector, self.detector_name, img)
            
            for face in faces:
                face_image = face[0]
                (x, y, w, h) = face[1]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face 
                cts=time.time()
                name=self.recognition(detected_face,False)
                if name!=new_name:
                    new_name=name
                    cv2.imwrite('test.png',detected_face)
                    image_file_descriptor = open('test.png', 'rb')
                    files = {'personImg': ('test.png',image_file_descriptor,'image/png')}
                    plate_data={'cameraid':'cam1','name':name,'time_in':time.ctime()}
                    res1=requests.post(self.post_url,data=plate_data,files=files)
                    image_file_descriptor.close()
                    os.remove('test.png')
                #     #post api with image upload of the unkown person
                cv2.putText(img, f"Name:{name}{time.time()-cts:.4f} ", (x, y + h + 30), fontface, 0.6, fontcolor, 2)
                print(name)
            ct=time.time()
            fps=fs/(ct-pt)
            cv2.putText(img, f"FPS:{round(fps)} ", (20,50), fontface, 0.6, fontcolor, 2)
            cv2.imshow('Video', img)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            # cv2.waitKey(1)
        cam.release()
    def out_streamer(self):
        cam = cv2.VideoCapture(self.cam_id)
        pt=time.time()
        fs=0
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontcolor = (0, 255, 0)
        while True:
            res,img = cam.read()
            if not res:
                break
            fs+=1
            faces = FaceDetector.detect_faces(self.detector, self.detector_name, img)
            for face in faces:
                face_image = face[0]
                (x, y, w, h) = face[1]
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face 
                cts=time.time()
                name=self.recognition(detected_face,False)
                if name!='unkown':
                    plate_data={'cameraid':'cam1','time_out':time.ctime()}
                    res1=requests.put(self.put_url+name,data=plate_data)   
                    #post api to add the registered user with exit time
                cv2.putText(img, f"Name:{name}{time.time()-cts:.4f} ", (x, y + h + 30), fontface, 0.6, fontcolor, 2)
                print(name)
            ct=time.time()
            fps=fs/(ct-pt)
            cv2.putText(img, f"FPS:{round(fps)} ", (20,50), fontface, 0.6, fontcolor, 2)
            cv2.imshow('Video', img)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            # cv2.waitKey(1)
        cam.release()
    

input_path="./input.mp4"
db_path="./database"
model_name='VGG-Face' #"VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"
model_metric='cosine'
obj=FaceRecognition(input_path,db_path,model_name,model_metric)
# obj.in_streamer()
obj.out_streamer()



                    # plate_image.save(buf, format='JPEG')
                    # image_file_descriptor = buf.getvalue()
                    # files = {'plateImage': image_file_descriptor}
                    # plate_data={'cameraid':'sdfs89769','plate':plate_text,'time_in':time.ctime()}
                    # res1=requests.post(anpr_model.post_url,data=plate_data,files=files)