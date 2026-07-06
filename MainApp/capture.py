import numpy as np
import os
import json
import base64

import cv2


face_detection_score_threshold = 0.8
face_detection_nms_threshold = 0.3
face_detection_top_k = 5000
two_face_match_threshold = 0.30


net = cv2.FaceDetectorYN.create(
    "Files/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    face_detection_score_threshold,
    face_detection_nms_threshold,
    face_detection_top_k,
)

recognizer = cv2.FaceRecognizerSF.create(
    "Files/face_recognition_sface_2021dec.onnx",
    "",
)


def encode_image_to_base64(image_np):
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")


class Capture_Faces:
    def __init__(self):
        print("Created")
        self.two_face_match_threshold = two_face_match_threshold
        path = "faces"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        with open("faces/data.json", "a+") as f:
            f.seek(0)
            if f.read():
                f.seek(0)
                face_data = json.load(f)
            else:
                face_data = {"name_key": [], "img_data": {}}
        self.face_data = face_data
        assert len(self.face_data["img_data"].keys()) == len(self.face_data["name_key"])

        if os.path.exists("faces/face_embeddings_sface.npz"):
            data = np.load("faces/face_embeddings_sface.npz")
            self.embedding_faces = {i: data[i] for i in data.files}
        else:
            self.embedding_faces = {}
        if len(self.embedding_faces.keys()) != len(self.face_data["name_key"]):
            self.extract_emb()

    def face_check(self, img):
        h, w = img.shape[:2]
        net.setInputSize((w, h))
        _, faces = net.detect(img)

        if faces is None or len(faces) == 0:
            raise Exception("No face found")
        return faces

    def embedding(self, img, face):
        return recognizer.feature(recognizer.alignCrop(img, face))

    def similarity(self, emb1, emb2):
        sim = recognizer.match(emb1, emb2, 0)
        return sim >= self.two_face_match_threshold

    def embedding_with_crop(self, face, img):
        x1, y1, x2, y2 = face[:4].astype(int)
        x2 += x1
        y2 += y1
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return False

        return self.embedding(img, face)

    def face_comparison(self, original_emb, ref_emb):
        return self.similarity(original_emb, ref_emb)

    def extract_emb(self):
        self.embedding_faces = dict()
        for i in range(len(self.face_data["name_key"])):
            if (
                str(i) not in self.face_data["img_data"]
                or not self.face_data["img_data"][str(i)]
            ):
                raise ValueError(
                    f"Corrupted data for name: {self.face_data['name_key'][i]}"
                )
            ref_img = cv2.imread(self.face_data["img_data"][str(i)])
            if ref_img is None:
                raise ValueError(
                    f"Could not read image for name: {self.face_data['name_key'][i]}"
                )
            faces = self.face_check(ref_img)
            best_face = max(faces, key=lambda f: f[-1])
            self.embedding_faces[str(i)] = self.embedding(ref_img, best_face)
        np.savez("faces/face_embeddings_sface.npz", **self.embedding_faces)

    def update_emb(self):
        for i in range(len(self.face_data["name_key"])):
            if (
                str(i) not in self.face_data["img_data"]
                or not self.face_data["img_data"][str(i)]
            ):
                raise ValueError(
                    f"Corrupted data for name: {self.face_data['name_key'][i]}"
                )
            if str(i) in self.embedding_faces:
                continue
            ref_img = cv2.imread(self.face_data["img_data"][str(i)])
            if ref_img is None:
                raise ValueError(
                    f"Could not read image for name: {self.face_data['name_key'][i]}"
                )
            faces = self.face_check(ref_img)
            best_face = max(faces, key=lambda f: f[-1])
            self.embedding_faces[str(i)] = self.embedding(ref_img, best_face)
        np.savez("faces/face_embeddings_sface.npz", **self.embedding_faces)

    def extract_eligible_faces(self, frame):
        assert len(self.face_data["img_data"].keys()) == len(self.face_data["name_key"])
        assert len(self.embedding_faces.keys()) == len(self.face_data["name_key"])
        eligible_faces = []
        matching = False
        try:
            faces = self.face_check(frame)
        except Exception as e:
            if str(e) == "No face found":
                return eligible_faces, matching
            else:
                raise Exception(e)

        if len(self.face_data["name_key"]) == 0:
            for face in faces:
                x1, y1, x2, y2 = face[:4].astype(int)
                x2 += x1
                y2 += y1
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                eligible_faces.append(encode_image_to_base64(frame[y1:y2, x1:x2]))
        else:
            for face in faces:
                x1, y1, x2, y2 = face[:4].astype(int)
                x2 += x1
                y2 += y1
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_match = False
                frame_face = self.embedding_with_crop(face, frame)
                for i in range(len(self.face_data["name_key"])):
                    if (
                        str(i) not in self.face_data["img_data"]
                        or not self.face_data["img_data"][str(i)]
                    ):
                        raise ValueError(
                            f"Corrupted data for name: {self.face_data['name_key'][i]}"
                        )
                    ref_emb = self.embedding_faces[str(i)]
                    if self.face_comparison(frame_face, ref_emb):
                        face_match = True
                if not (face_match):
                    eligible_faces.append(encode_image_to_base64(frame[y1:y2, x1:x2]))
        if len(faces) > 0 and len(eligible_faces) == 0:
            matching = True

        return eligible_faces, matching

    def video(self, frame):
        assert len(self.face_data["img_data"].keys()) == len(self.face_data["name_key"])
        assert len(self.embedding_faces.keys()) == len(self.face_data["name_key"])
        try:
            faces = self.face_check(frame)
        except Exception as e:
            if str(e) == "No face found":
                return encode_image_to_base64(frame)
            else:
                raise Exception(e)

        output_frame = frame.copy()

        for face in faces:
            frame_face = self.embedding_with_crop(face, frame)
            for i in range(len(self.face_data["name_key"])):
                if (
                    str(i) not in self.face_data["img_data"]
                    or not self.face_data["img_data"][str(i)]
                ):
                    raise ValueError(
                        f"Corrupted data for name: {self.face_data['name_key'][i]}"
                    )

                ref_emb = self.embedding_faces[str(i)]

                if self.face_comparison(frame_face, ref_emb):
                    x1, y1, x2, y2 = face[:4].astype(int)
                    x2 += x1
                    y2 += y1
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        self.face_data["name_key"][i], font, font_scale, thickness
                    )
                    text_x = x1 + (x2 - x1 - text_width) // 2
                    text_y = y2 + text_height + 5

                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        output_frame,
                        self.face_data["name_key"][i],
                        (text_x, text_y),
                        font,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                    )

        return encode_image_to_base64(output_frame)


def save_faces(names, faces):
    path = "faces"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    with open("faces/data.json", "a+") as f:
        f.seek(0)
        if f.read():
            f.seek(0)
            face_data = json.load(f)
        else:
            face_data = {"name_key": [], "img_data": {}}

    assert len(names) == len(faces)

    index = len(face_data["name_key"]) + 1
    for i in range(0, len(names)):
        face_data["name_key"].append(names[i])
        face_data["img_data"][
            str(len(face_data["name_key"]) - 1)
        ] = f"faces/image{index}.jpg"
        img = base64.b64decode(faces[i])
        with open("faces/data.json", "w") as f:
            json.dump(face_data, f)
        with open(f"faces/image{index}.jpg", "wb") as f:
            f.write(img)
        index += 1
