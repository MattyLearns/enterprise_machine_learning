from flask import Flask, config, render_template, request, redirect, url_for, send_from_directory
from matplotlib.pyplot import table
from pandas import options
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import time
import cv2
import pandas as pd

import grpc
from grpc.beta import implementations

# Import prediction service functions from TF-Serving API
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from utils import label_map_util
from utils import visualization_utils as viz_utils
from core.standard_fields import DetectionResultFields as dt_fields

sys.path.append("..")
tf.get_logger(). setLevel('ERROR')

# labels file for the 90 class model
PATH_TO_LABELS = "./data/mscoco_label_map.pbtxt"
# number of classes to classify
NUM_CLASSES = 90

# maps the index number and category names of the classes
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# initailise Flask instance and configure parameters
app = Flask(__name__)

# configure uploads folder with the app
app.config['UPLOAD_FOLDER'] = 'uploads/'
# file extensions allowed to upload
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


# function to check for allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# this function creates interface to communicate with tensorflow serving
def get_stub(host='127.0.0.1', port='8500'):
    channel = grpc.insecure_channel('127.0.0.1:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


# convert image into numpy array of the shape height, width, channel
def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# tensorlfow requires the input to be a tensor
# convert numpy array to tensorflow tensor
def load_input_tensor(input_image):
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    return tensor


# function to perform inference on the image and draw bounding boxes
# it also returns the class names with their accuracy
def inference(frame, stub, model_name='detector'):
    # Add the RPC command here
    # Call tensorflow server
    # channel = grpc.insecure_channel('localhost:8500')
    channel = grpc.insecure_channel('localhost:8500', options=(('grpc.enable_http_proxy',0),))
    print("Channel: ", channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print('Stub: ', stub)
    request = predict_pb2.PredictRequest()
    print('Request: ', request)
    request.model_spec.name = 'detector'
    
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_im)
    input_tensor = load_input_tensor(image)
    request.inputs['input_tensor'].CopyFrom(input_tensor)

    result = stub.Predict(request, 60.0)

    # load image into numpy array
    image_np = load_image_into_numpy_array(image)
    # Copy of the original image_np is created and passed to the function
    # Both original and copy will be saved
    image_np_with_detections = image_np.copy()

    # the classes, bounding boxes, and accuracy scores are extracted 
    # and stored in a dictionary
    output_dict = {}
    output_dict['detection_classes'] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8)
    output_dict['detection_boxes'] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict['detection_scores'] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)

    # method to draw bounding boxes on the image
    frame = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

    # the names of the classes along with their accuracy are extracted and 
    # stored in the variable detected_objects
    detected_objects = [(category_index.get(value)['name'], 
                        output_dict['detection_scores'][index]) 
                        for index, value in enumerate(output_dict['detection_classes'])
                        if output_dict['detection_scores'][index] > 0.3]
    
    return frame, image_np, detected_objects

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('result', filename=filename))


@app.route('/results/<filename>')
def result(filename):
    detected_obj = uploaded_file(filename)
    # detected objects sets is converted to dataframe
    df = pd.DataFrame(detected_obj, columns=['Objects', 'Accuracy'])
    # dataframe is converted to html to be displayed on the website
    df_html = df.to_html()
    # return the result html file
    return render_template('results.html', filename=filename, dataframe=df_html)


# @app.route('/uploads/<filename>')
def uploaded_file(filename):
    # This function takes the uploaded image and pass it to the inference function
    # It also saves the inferenced image and original image into their respective folders
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)

    stub = get_stub()

    for image_path in TEST_IMAGE_PATHS:
        img_org = Image.open(image_path)
        # image_np = np.array(Image.open(image_path))
        image_np = np.array(img_org)
        # image_np is passed onto the inference function which returns inferenced and 
        # original image and detected objects with their accuracy
        image_np_inferenced, image_np_original, detected_obj = inference(image_np, stub)
        # Inferenced image is converted from array to image and saved in uploads folder
        im = Image.fromarray(image_np_inferenced)
        im.save('uploads/' + filename)
        im.save('static/uploads/' + filename)
        # original image is saved in the originals folder
        img_org.save('originals/' + filename)
        img_org.save('static/originals/' + filename)

    return detected_obj


@app.route('/donate')
def donate():
    # return the donate html file
    return render_template('donate.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
