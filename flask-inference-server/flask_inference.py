from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import torch
import torchvision



#Hazard_vest
from tracking.yolov4.load_model import Loadv4Model
from tracking.yolov4.detect import Detect


app = Flask(__name__)

image_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
model_file = 'model'
model = torch.jit.load(model_file)
model.eval()


@app.route("/yolo", methods=['POST'])
def lambda_handler():
    #print(request.data)
    image_bytes = request.data  # .encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    image = image.resize((28, 28))

    probabilities = model.forward(image_transforms(np.array(image)).reshape(-1, 1, 28, 28))
    label = torch.argmax(probabilities).item()

    ret = {"predicted_label": label}
    return jsonify(ret)


#hazard model


yolov4tiny_model = Loadv4Model().load_model()
yolov4tiny_model_detect_obj = Detect(yolov4tiny_model)

@app.route("/hazard", methods = ['POST'])
def hazard_model():
    #make into frames format
    image_bytes = request.data  # .encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='L')
    #image = image.resize((28, 28))


    res = yolov4tiny_model_detect_obj.detect(image)

    label = torch.argmax(res).item()

    ret = {"predicted_label": label}
    return jsonify(ret)

@app.route("/crowd_density", methods = ['POST'])
def crowd_density_model():
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
