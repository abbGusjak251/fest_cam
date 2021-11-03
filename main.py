# Import modules
import cv2
from datetime import datetime
import time
import os
import sys

# Define face detection cascade path
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Start video capture for face detection
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

def search_for_faces(path):
    global faceCascade

    # Read image and define color gray
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) > 0:
        # Log the number of faces that were found
        print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(path, image)
    else:
        print("Found no faces")
        os.remove(path)

def generate_image_name():
    return f"./images/{str(datetime.now()).replace(' ', '').replace(':', '').replace('.', '')}.jpg"

def save_image(frame):
    path = generate_image_name()
    cv2.imwrite(path, frame)
    return path

counter = 0

while True:
    try:
        try:
            ret, frame = cam.read()
            path = save_image(frame)
            search_for_faces(path)
        except Exception as e:
            print(e)
        counter += 1
        if counter > 10:
            break
        time.sleep(4)
    except KeyboardInterrupt:
        exit()
cam.release()
cv2.destroyAllWindows()