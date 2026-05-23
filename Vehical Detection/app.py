from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# 🔹 Model 1: Vehicle Detection
vehicle_model = YOLO("yolov8n.pt")

# 🔹 Model 2: Accident Classification
accident_model = YOLO("runs/classify/damage_model-2/weights/best.pt")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]

@app.route("/", methods=["GET", "POST"])
def index():

    detected_names = []
    filename = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename != "":
            
            # ensure folder exists
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            filename = file.filename

            img = cv2.imread(filepath)

            # safety check
            if img is None:
                return "Error loading image"

            results = vehicle_model(filepath)

            for box in results[0].boxes:

                class_id = int(box.cls)
                name = vehicle_model.names[class_id]
                conf = float(box.conf)

                if name in vehicle_classes and conf > 0.5:

                    if name == "car":

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = img[y1:y2, x1:x2]

                        if crop is None or crop.size == 0:
                            detected_names.append("car")
                            continue

                        cls_result = accident_model(crop)

                        pred = cls_result[0].probs.top1
                        label = accident_model.names[pred]

                        print("Predicted:", label)

                        if label == "accident":
                            detected_names.append("accident_car")
                        else:
                            detected_names.append("car")

                    else:
                        detected_names.append(name)

    return render_template(
        "index.html",
        names=detected_names,
        filename=filename
    )

if __name__ == "__main__":
    app.run(debug=True)