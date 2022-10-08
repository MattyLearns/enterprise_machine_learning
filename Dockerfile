FROM ubuntu
FROM python:3

# Add project
ADD FlaskObjectDetection /FlaskObjectDetection

# Download and install dependency
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install Flask
RUN pip install tensorflow
RUN pip install tensorflow-serving-api
RUN pip install Flask-WTF
RUN pip install numpy
RUN pip install opencv-python
RUN pip install pafy
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install grpcio
RUN pip install pandas

# Step 3: Tell the image what to do when it starts as container
WORKDIR "./FlaskObjectDetection"
CMD ["python", "./app.py"]
