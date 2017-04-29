import glob
import time
import cv2
import numpy as np
from vehicle.features import extract
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


car_files = glob.glob('./data/vehicles/*/*.png')
non_car_files = glob.glob('./data/non-vehicles/*/*.png')
print('Vehicles: %d' % len(car_files))
print('Non-vehicles: %d' % len(non_car_files))

cars = map(lambda x: extract(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2HSV)), car_files)
negatives = map(lambda x: extract(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2HSV)), non_car_files)

ts = time.time()
x_data_cars = np.squeeze(np.array(list(cars)))
x_data_negatives = np.squeeze(np.array(list(negatives)))
print('Extracted features in %.2f seconds' % (time.time() - ts))
print('Feature vector length:', len(x_data_cars[0]))

# combine and scale
x_data = np.vstack((x_data_cars, x_data_negatives)).astype(np.float32)
scaler = StandardScaler().fit(x_data)
x_data_scaled = scaler.transform(x_data)

# labels
y = np.hstack((np.ones(len(x_data_cars)), np.zeros(len(x_data_negatives))))

# split train test
x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y, test_size=0.1, stratify=y)

# train classifier
svc = LinearSVC()
# Check the training time for the SVC
ts = time.time()
svc.fit(x_train, y_train)
print('Trained classifier in %.2f seconds' % (time.time() - ts))

# test classifier
print('Test Accuracy of %s is %.4f' % (svc, svc.score(x_test, y_test)))

# saving
joblib.dump(scaler, './data/scaler.clf')
joblib.dump(svc, './data/classifier.clf')
