import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import tensorflow as tf
import numpy as np
import mtcnn
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import sys
from moviepy.editor import VideoFileClip
from moviepy.editor import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



from mtcnn.mtcnn import MTCNN
mcnn = MTCNN()



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def vid_to_frames(filename, frames):
    cap = cv2.VideoCapture(filename)
    images = []
    i = 0
    print('Processing Frames')
    printProgressBar(0, frames, prefix = 'Progress: ', suffix = 'Frames Processed', length = 20)
    while(cap.isOpened()):
            i += 1
            ret, frame = cap.read()
            if ret == True:
                images.append(frame)
                cv2.imshow('Capturing',frame)
                if cv2.waitKey(41) & 0xFF == ord('q'):
                    break
            else:
                break
            printProgressBar(i, frames, prefix = 'Progress: ', suffix = 'Frames Processed', length = 20)
    cap.release()
    cv2.destroyAllWindows()
    return images

def videolenframes(filename):
    clip = VideoFileClip(filename)
    time = clip.duration
    frames = time * 23.85
    return frames


def get_faces_and_boxes(img, size = (160, 160)):
    imarray = img
    boxes = mcnn.detect_faces(imarray)
    faces = []

    for box in boxes:
        x, y, w, h = box['box']
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + w, y1 + h
        face = imarray[y1:y2, x1:x2]
        face = cv2.resize(face, size)
        faces.append((abs(x), abs(y), w, h, face))
    return faces


def img_with_boxes(img, faces):
    image = img
    for face in faces:
        x, y, w, h, _ = face
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color = (0, 0, 255), thickness = -1)
    return image


def censored_video(images, destination):
    shape = images.shape[1:-1]
    h, w = shape
    print('Video resolution is ', shape)
    frames = images.shape[0]
    video = cv2.VideoWriter(destination, cv2.VideoWriter_fourcc(*'DIVX'), 24, (w, h))
    i = 0
    print('Anonymising Video')
    printProgressBar(i, frames, prefix = 'Progress: ', suffix = 'Frames Processed', length = 20)
    for image in images:
        i += 1
        faces = get_faces_and_boxes(image)
        image = img_with_boxes(image, faces)
        video.write(image)
        printProgressBar(i, frames, prefix = 'Progress: ', suffix = 'Frames Processed', length = 20)
    video.release()

def main():
    frames = videolenframes(sys.argv[1])
    images = vid_to_frames(sys.argv[1], frames)
    images = np.array(images)
    np.save('Processed/temp.npy', images)
    print()
    print('Video processed, ', len(images), ' frames extracted')
    images = np.load('Processed/temp.npy')
    tik = time.time()
    censored_video(images, sys.argv[2])
    tok = time.time()
    print('Anonymisation took ', round(tok - tik, 2), ' seconds')
    print('Video saved at ', sys.argv[2])

if(__name__ == '__main__'):
    main()











