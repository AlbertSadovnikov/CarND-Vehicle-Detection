from vehicle.pipeline import process
import numpy as np
from sklearn.externals import joblib
import cv2
import time


wd = np.load('./data/windows.npz')
windows = wd['windows']
weightmap = wd['weightmap']

scaler = joblib.load('./data/scaler.clf')
clf = joblib.load('./data/classifier.clf')

frame = cv2.imread('./test_images/test6.jpg')

ts = time.time()
result = process(frame, windows, scaler, clf, weightmap)
print('One frame processing took: %.2f' % (time.time() - ts))

cv2.namedWindow('sample', cv2.WINDOW_NORMAL)
cv2.imshow('sample', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
