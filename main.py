
from flask import Flask,request,jsonify
from tensorflow.keras.models import load_model 
 # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

import requests
import urllib.request
import cv2
import ssl

import os


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Vegetable Disease Classifier running Correctly'


# @app.route("/vegeteable-disease-classifier",methods = ['POST'])
# def classifyWaste():  
 
#   # Check if JSON data is provided in the request
#   if not request.json or 'url' not in request.json:
#     return jsonify(error="No URL provided in JSON data"), 400
    
#   #1 get url
#   print(request,"req")
#   image_url = request.json["url"]
#   # image_url = "https://utfs.io/f/813a813a-521b-4d4d-92c3-d1d46d2de812-2fz.jpg"
#   print("get image url") 

#   try:
#     #2 Read image from cloud
#     requestImg = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
#     response=urllib.request.urlopen(requestImg)
#     # image = urllib.URLopener()
#     print(response,"open url img")

#     #3 convert response into array using numpy
#     img_array=np.array(bytearray(response.read()), dtype=np.uint8)

#     #4 Decode img_array into RGB 
#     img=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
#     print("img")

#     #5 Resize the image to 240x240
#     resized_img = cv2.resize(img, (224, 224))
#     print("resized")

#     # print(imgpath,"classify")
#     # Disable scientific notation for clarity
#     np.set_printoptions(suppress=True)

#     # Load the labels
#     labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.txt")
#     print(labels_path)
#     class_names = open(labels_path, "r").readlines()
#     print("labels loaded")

#     # Load the model
#     model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keras_model.h5")
#     print(model_path)
#     model = load_model(model_path, compile=False)
#     print("model loaded")

    

#     # Create the array of the right shape to feed into the keras model
#     # The 'length' or number of images you can put into the array is
#     # determined by the first position in the shape tuple, in this case 1
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#     print("data loaded")

#     #6 Normalize the image
#     normalized_image_array = (resized_img.astype(np.float32) / 127.5) - 1
#     print("normalized")

#     #7 Load the image into the array
#     data[0] = normalized_image_array
#     print("data[0]")

#     #8 Predicts the model
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]
#     print("all done")

#     #9 Print prediction and confidence score
#     result={
#       "Class": (class_name[2:].strip("\n")),
#       "Confidence Score": str(confidence_score)
#     }
#     print(result,"result")
#     # print("Class:", class_name[2:], end="")
#     # print("Confidence Score:", confidence_score)
    
#     return jsonify(result),200

#   except:
#     return jsonify(error="Problem in reading Image from Provided Url"),404




# file
@app.route("/asfile-vegeteable-disease-classifier", methods=['POST'])
def classify_waste_asFile():  
    # Check if an image file is provided in the request
    if 'file' not in request.files:
        return jsonify(error="No file provided"), 400

    # Get the image file from the request
    image_file = request.files['file']
    
    if image_file.filename == '':
        return jsonify(error="No selected file"), 400

    try:
        # Open the image using PIL
        image = Image.open(image_file)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(image)
        
        # Convert the image to RGB if it is not already
        if image.mode != "RGB":
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        print("img")

        # Resize the image to 224x224 (already done by ImageOps.fit)
        resized_img = cv2.resize(img_array, (224, 224))
        print("resized")

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the labels
        labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.txt")
        print(labels_path)
        class_names = open(labels_path, "r").readlines()
        print("labels loaded")

        # Load the model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keras_model.h5")
        print(model_path)
        model = load_model(model_path, compile=False)
        print("model loaded")

        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        print("data loaded")

        # Normalize the image
        normalized_image_array = (resized_img.astype(np.float32) / 127.5) - 1
        print("normalized")

        # Load the image into the array
        data[0] = normalized_image_array
        print("data[0]")

        # Predict using the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        print("all done")

        # Return the prediction and confidence score
        result = {
            "Class": class_name[2:].strip("\n"),
            "Confidence Score": str(confidence_score)
        }
        print(result, "result")
        
        return jsonify(result), 200

    except Exception as e:
        print(str(e))
        return jsonify(error="Problem in processing the image file"), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')