from moviepy.editor import VideoFileClip
from vehicle.pipeline import process
import numpy as np
from sklearn.externals import joblib
from argparse import ArgumentParser
import cv2

wd = np.load('./data/windows.npz')
windows = wd['windows']
weightmap = wd['weightmap']
scaler = joblib.load('./data/scaler.clf')
clf = joblib.load('./data/classifier.clf')

winname = ''

if winname != '':
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)


def process_frame(frame):
    # moviepy gets rgb, process needs bgr, and return bgr
    result = process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), windows, scaler, clf, weightmap, winname=winname)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

parser = ArgumentParser(description='CarND Vehicle detection')
parser.add_argument('--video', help='Path to video file', dest='video_filename', required=True)
parser.add_argument('--output', help='Path to output video file', dest='out_filename', required=True)
args = parser.parse_args()
clip = VideoFileClip(args.video_filename)
output_clip = clip.fl_image(process_frame)
output_clip.write_videofile(args.out_filename, audio=False, progress_bar=True)
