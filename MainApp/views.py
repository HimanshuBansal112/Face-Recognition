import base64
import json
from django.core.files.base import ContentFile
from django.shortcuts import render
import numpy as np
import cv2
from .capture import Capture_Faces, save_faces
from django.http import JsonResponse

capture_faces = Capture_Faces()


def index(request):
    return render(request, "MainApp/index.html")


def store(request):
    global capture_faces
    if request.method == "POST":
        action = request.POST.get("action")
        image = request.POST.get("image")
        if (action == "process") and image:
            valid = False
            if image:
                _, imgstr = image.split(";base64,")
                img_data = base64.b64decode(imgstr)
                if img_data:
                    np_arr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    valid = True

                    output, matching = capture_faces.extract_eligible_faces(frame)
                    return render(
                        request,
                        "MainApp/store.html",
                        {
                            "data": {
                                "uploaded": valid,
                                "output": output,
                                "matching": matching,
                                "finished": False,
                            }
                        },
                    )

            return render(
                request,
                "MainApp/store.html",
                {"data": {"uploaded": valid, "finished": False}},
            )
        elif action == "store":
            names = json.loads(request.POST.get("face_names"))
            faces = json.loads(request.POST.get("faces"))
            save_faces(names[1:], faces[1:])
            capture_faces = Capture_Faces()
            capture_faces.update_emb()
            return render(
                request,
                "MainApp/store.html",
                {"data": {"uploaded": True, "finished": True}},
            )
    return render(request, "MainApp/store.html", {"data": ""})


def display(request):
    return render(request, "MainApp/display.html")


def process(request):
    if request.method == "POST":
        data = json.loads(request.body)
        imgstr = data.get("image")

        if imgstr:
            img_data = base64.b64decode(imgstr)
            if img_data:
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                return JsonResponse({"output": capture_faces.video(frame)})
        return JsonResponse({}, status=400)
