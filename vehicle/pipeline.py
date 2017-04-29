import cv2
from vehicle.image import downscale, patch, im_rect
from vehicle.features import extract
from vehicle.plotting import imagesc
from scipy.ndimage.measurements import label
from skimage.morphology import opening, disk
import numpy as np
import string
import random


PATCH_STEP = (16, 16)
DEBUG = False


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def process(image, windows, scaler, clf, weightmap, winname=''):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    heatmap = np.zeros((hsv.shape[:2]), np.float32)
    for w in windows:
        ds = downscale(hsv, w, 128)
        if winname != '':
            cv2.waitKey(1)
        for p, px, py in patch(ds, (64, 64), PATCH_STEP):
            features = extract(p)
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)
            has_car = clf.predict(features_scaled)
            r = im_rect(px, py, (64, 64), w, 128)
            if has_car:
                heatmap[r[1]:r[3], r[0]:r[2]] += 1
                if DEBUG:
                    cv2.rectangle(image, tuple(r[:2]), tuple(r[2:]), (255, 0, 0), 1)
                    cv2.imwrite('output_images/boxes.jpg', image)

    if DEBUG:
        heatmap_sc = imagesc(heatmap)
        cv2.imwrite('output_images/heatmap.jpg', heatmap_sc)
    result = heatmap / weightmap
    if DEBUG:
        heatmap_nr = imagesc(result)
        cv2.imwrite('output_images/heatmap_norm.jpg', heatmap_nr)

    labels = label(opening(result > 0.10, disk(10)))
    draw_labeled_bboxes(image, labels)
    if winname != '':
        cv2.imshow(winname, image)
        cv2.waitKey(1)
    return image


def random_name(length=12):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length)) + '.png'


def mining(winname, image, windows, scaler, clf):
    print('new frame')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for w in windows:
        ds = downscale(hsv, w, 128)
        cv2.waitKey(1)
        for p, px, py in patch(ds, (64, 64), PATCH_STEP):
            features = extract(p)
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)
            has_car = clf.predict(features_scaled)
            if has_car:
                print('check sample')
                sample = cv2.cvtColor(p, cv2.COLOR_HSV2BGR)
                cv2.imshow(winname, sample)
                while True:
                    key = cv2.waitKey(0) & 0xff
                    if key == ord('p'):
                        cv2.imwrite('./data/vehicles/mining/' + random_name(), sample)
                        print('positive written')
                        break
                    elif key == ord('n'):
                        cv2.imwrite('./data/non-vehicles/mining/' + random_name(), sample)
                        print('negative written')
                        break
                    else:
                        print('skipped ')
                        break
    return

