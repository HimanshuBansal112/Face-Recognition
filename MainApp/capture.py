import numpy as np
import os
import json
import base64

import cv2
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from facenet_pytorch import InceptionResnetV1


net = cv2.dnn.readNetFromCaffe(
    'Files/deploy.prototxt', 
    'Files/res10_300x300_ssd_iter_140000.caffemodel'
)

def encode_image_to_base64(image_np):
    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')

class Capture_Faces:
    def __init__(self):
        print("Created")
        self.two_face_match_threshold = 0.6
        self.human_face_confidence = 0.5
        path="faces"
        isExist = os.path.exists(path)
        if not isExist:
           os.makedirs(path)
        with open('faces/data.json', "a+") as f:
            f.seek(0)
            if f.read():
              f.seek(0)
              face_data = json.load(f)
            else:
              face_data = {"name_key": [], "img_data":{}}
        self.face_data = face_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        assert len(self.face_data["img_data"].keys())==len(self.face_data["name_key"])
        
        if os.path.exists("faces/face_embeddings.pt"):
            self.embedding_faces = torch.load("faces/face_embeddings.pt", map_location=self.device)
        else:
            self.embedding_faces = {}
            torch.save(self.embedding_faces, "faces/face_embeddings.pt")
        #self.extract_emb()

    def face_check(self, img):
        faces=[]
        (h, w) = img.shape[:2]
        
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                   (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > self.human_face_confidence:  #Confidence that the face is of human
              box = detections[0, 0, i, 3:7] * [w, h, w, h]
              faces.append(box.astype("int"))
        if len(faces)==0:
          raise Exception("No face found")
        return faces

    def embedding(self, img):
        face = resize(img, [160, 160])
        face = face.float() / 255.0
        face = (face - 0.5) / 0.5
        face = face.unsqueeze(0).to(self.device)
        emb = self.embed_model(face).detach()
        emb = emb / emb.norm()
        return emb.cpu()
    
    def similarity(self, emb1, emb2):
        sim = torch.nn.functional.cosine_similarity(emb1, emb2)
        return sim > self.two_face_match_threshold #It is being used to ensure that two faces match
    
    def embedding_with_crop(self, face, tensor_img):
        x1,y1,x2,y2 = face
        _, H, W = tensor_img.shape
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return False #False is 0 so it will work like "no face found"
            
        emb = self.embedding(tensor_img[:,y1:y2,x1:x2])
        return emb
    
    def face_comparison(self, original_emb, ref_emb):
        return self.similarity(original_emb, ref_emb)
    
    def extract_emb(self):
        self.embedding_faces = dict()
        for i in range(len(self.face_data["name_key"])):
            if str(i) not in self.face_data["img_data"] or not self.face_data["img_data"][str(i)]:
                raise ValueError(f"Corrupted data for name: {self.face_data['name_key'][i]}")
            ref_tensor_img = read_image(self.face_data["img_data"][str(i)])
            self.embedding_faces[str(i)] = self.embedding(ref_tensor_img)
        torch.save(self.embedding_faces, "faces/face_embeddings.pt")
    
    def update_emb(self):
        for i in range(len(self.face_data["name_key"])):
            if str(i) not in self.face_data["img_data"] or not self.face_data["img_data"][str(i)]:
                raise ValueError(f"Corrupted data for name: {self.face_data['name_key'][i]}")
            if str(i) in self.embedding_faces:
                continue
            ref_tensor_img = read_image(self.face_data["img_data"][str(i)])
            self.embedding_faces[str(i)] = self.embedding(ref_tensor_img)
        torch.save(self.embedding_faces, "faces/face_embeddings.pt")
    
    def extract_eligible_faces(self, frame):
        assert len(self.face_data["img_data"].keys())==len(self.face_data["name_key"])
        assert len(self.embedding_faces.keys())==len(self.face_data["name_key"])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        eligible_faces = []
        matching = False
        try:
            faces = self.face_check(frame)
        except Exception as e:
            if str(e) == "No face found":
                return eligible_faces, matching
            else:
                raise Exception(e)
        
        if len(self.face_data["name_key"])==0:
            for face in faces:
                x1,y1,x2,y2 = face
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                eligible_faces.append(encode_image_to_base64(frame[y1:y2, x1:x2]))
        else:
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
            frame_tensor = frame_tensor.to(torch.uint8)
            for face in faces:
                x1,y1,x2,y2 = face
                face_match = False
                frame_face = self.embedding_with_crop(face, frame_tensor)
                for i in range(len(self.face_data["name_key"])):
                    if str(i) not in self.face_data["img_data"] or not self.face_data["img_data"][str(i)]:
                        raise ValueError(f"Corrupted data for name: {self.face_data['name_key'][i]}")
                    ref_emb = self.embedding_faces[str(i)]
                    if self.face_comparison(frame_face, ref_emb):
                        face_match = True
                if not(face_match):
                    eligible_faces.append(encode_image_to_base64(frame[y1:y2, x1:x2]))
        if len(faces)>0 and len(eligible_faces)==0:
            matching = True
        
        return eligible_faces, matching
        
    def video(self, frame):
        assert len(self.face_data["img_data"].keys())==len(self.face_data["name_key"])
        assert len(self.embedding_faces.keys())==len(self.face_data["name_key"])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            faces = self.face_check(frame)
        except Exception as e:
            if str(e) == "No face found":
                return encode_image_to_base64(frame)
            else:
                raise Exception(e)
        
        output_frame = frame.copy()
        
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
        frame_tensor = frame_tensor.to(torch.uint8)
        
        for face in faces:
            frame_face = self.embedding_with_crop(face, frame_tensor)
            for i in range(len(self.face_data["name_key"])):
                if str(i) not in self.face_data["img_data"] or not self.face_data["img_data"][str(i)]:
                    raise ValueError(f"Corrupted data for name: {self.face_data['name_key'][i]}")
                
                ref_emb = self.embedding_faces[str(i)]
                
                if self.face_comparison(frame_face, ref_emb):
                    x1,y1,x2,y2 = face
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(self.face_data["name_key"][i], font, font_scale, thickness)
                    text_x = x1 + (x2 - x1 - text_width) // 2
                    text_y = y2 + text_height + 5
                    
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_frame, self.face_data["name_key"][i], (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
                    
        return encode_image_to_base64(output_frame)

def save_faces(names, faces):
    path="faces"
    isExist = os.path.exists(path)
    if not isExist:
       os.makedirs(path)
    with open('faces/data.json', "a+") as f:
        f.seek(0)
        if f.read():
          f.seek(0)
          face_data = json.load(f)
        else:
          face_data = {"name_key": [], "img_data":{}}
    
    assert len(names)==len(faces)
    
    index = len(face_data["name_key"]) + 1
    for i in range(0,len(names)):
        face_data["name_key"].append(names[i])
        face_data["img_data"][str(len(face_data["name_key"]) - 1)] = f"faces/image{index}.jpg"
        img = base64.b64decode(faces[i])
        with open('faces/data.json', "w") as f:
            json.dump(face_data, f)
        with open(f"faces/image{index}.jpg", "wb") as f:
            f.write(img)
        index += 1