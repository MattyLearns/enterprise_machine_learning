# Enterprise Machine Learning solution for object detection
This is an object detection flask web app running with tenssorflow server. It can classify 90 different classes of common objects. The algorithm Faster-RCNN has been trained using TensorFlow Object Detection API.

This file contains instructions on executing the commands to run object detection web app. 
The steps are given below.

1. Command to build docker file for the object detection web app with the name detector-app.
sudo docker build -t detector-app .

2. Command to create netwrok binding and run Tensorflow server with GPU support.
sudo docker run --gpus all -p 8500:8500 --name detector --mount type=bind,source=/home/msc1/Desktop/Labs/Semester_2/7147COMP/coursework/saved_model,target=/models/detector -e MODEL_NAME=detector -t tensorflow/serving:latest-gpu

3. Command to run web app directly, open another terminal and go to FlaskObjectDetection 
directory. Run the following command.
python app.py

4. Command to run the web app docker image created in step 1.
sudo docker run -p 127.0.0.1:5000:5000 detector-app

Some screenshots from the webapp is given below.
![webapp_home](https://user-images.githubusercontent.com/30217266/194730227-bceb3669-e0ac-4c5d-8d71-ef3d4d662599.png)
![webapp_result](https://user-images.githubusercontent.com/30217266/194730229-cf318dac-58b7-4296-a451-5035977ec9b6.png)
![webapp_result_2](https://user-images.githubusercontent.com/30217266/194730232-0c80106c-ef82-4026-9eb1-8509b15451a2.png)
