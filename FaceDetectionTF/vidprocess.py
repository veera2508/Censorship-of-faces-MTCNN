import cv2
import numpy as np
import sys

def vid_to_frames(filename):
    cap = cv2.VideoCapture(filename)
    images = []
    while(True):
            ret, frame = cap.read()
            if ret == True:
                images.append(frame)
                cv2.imshow('Capturing',frame)
                if cv2.waitKey(41) & 0xFF == ord('q'):
                    break
            else:
                break
    cap.release()
    cv2.destroyAllWindows()
    return images

def main():
    images = vid_to_frames(sys.argv[1])
    images = np.array(images)
    np.save('Processed/ec.npy', images)
    print(len(images))

if(__name__ == '__main__'):
    main()