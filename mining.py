from moviepy.editor import VideoFileClip
from vehicle.pipeline import mining
import numpy as np
from sklearn.externals import joblib
from argparse import ArgumentParser
import cv2

wd = np.load('./data/windows.npz')
windows = wd['windows']
scaler = joblib.load('./data/scaler.clf')
clf = joblib.load('./data/classifier.clf')
winname = 'mining'

cv2.namedWindow(winname, cv2.WINDOW_NORMAL)


def process_frame(frame):
    mining(winname, frame, windows, scaler, clf)
    return

parser = ArgumentParser(description='CarND Vehicle detection mining')
parser.add_argument('--video', help='Path to video file', dest='video_filename', required=True)
args = parser.parse_args()


cap = cv2.VideoCapture(args.video_filename)

skip_frames = 400

while cap.isOpened():
    ret, image = cap.read()
    if ret:
        if skip_frames == 0:
            process_frame(image)
            skip_frames = 10
        else:
            skip_frames -= 1

cap.release()
cv2.destroyAllWindows()
