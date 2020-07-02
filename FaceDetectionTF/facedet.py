# %%
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import cv2
import pandas
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Lambda
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mtcnn
tf.keras.backend.clear_session()
print(tf.__version__, mtcnn.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
%matplotlib inline

# %%
from mtcnn.mtcnn import MTCNN
from PIL import Image
mcnn = MTCNN()

# %%
def get_faces_and_boxes(img, size = (160, 160)):
    imarray = img


    boxes = mcnn.detect_faces(imarray)
    faces = []

    for box in boxes:
        x, y, w, h = box['box']
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + w, y1 + h
        face = imarray[y1:y2, x1:x2]
        face = Image.fromarray(face)
        face = face.resize(size)
        face = np.asarray(face)
        faces.append((abs(x), abs(y), w, h, face))
    
    return faces


# %%
def img_with_boxes(img, faces):
    image = img
    for face in faces:
        x, y, w, h, _ = face
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color = (0, 0, 255), thickness = -1)
    return image

# %%
def img_with_masks(filename, faces):
    image = np.asarray(Image.open(filename))
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for face in faces:
        x, y, w, h, _ = face
        mask = matplotlib.patches.Ellipse(xy = ((x + w/2), (y + h/2)), width = w, height = h, linewidth = 1, edgecolor = 'black', facecolor = 'black')
        ax.add_patch(mask)
    plt.axis('off')
    plt.show()

# %%
filename = 'Samples/sample2.JPG'
faces = get_faces_and_boxes(filename)

# %%
img_with_boxes(filename, faces)
img_with_masks(filename, faces)
# %%
def vid_to_frames(filename):
    cap = cv2.VideoCapture(filename)
    images = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        images.append(frame)
        cv2.imshow('Capturing',frame)
        if cv2.waitKey(41) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return images

# %%
images = vid_to_frames('Samples/rr360.mp4')

# %%
print(len(images))

# %%
def censored_video(images):
    video = cv2.VideoWriter('Samples/censored.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, (640, 360))
    i = 0
    for image in images:
        print('Working on frame: ', i)
        i += 1
        faces = get_faces_and_boxes(image)
        image = img_with_boxes(image, faces)
        b,g,r = cv2.split(image)
        image2 = cv2.merge([r,g,b])
        video.write(image)
        plt.imshow(image2)
        plt.axis('off')
        plt.show()
    video.release()

# %%
censored_video(images)

# %%
