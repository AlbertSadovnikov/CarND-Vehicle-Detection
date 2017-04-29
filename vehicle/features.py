import numpy as np
import cv2
from skimage.feature import hog


def extract(img):
    sf = bin_spatial(img)
    cf = color_hist(img)
    hfs = hog_features(img[:, :, 1])
    hfv = hog_features(img[:, :, 2])
    return np.hstack((sf, cf, hfs, hfv))


def hog_features(img, orient=9, pix_per_cell=16, cell_per_block=2, vis=False, feature_vec=True):
    features = hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=False,
                   visualise=vis,
                   feature_vector=feature_vec,
                   block_norm='L2')
    return features


def bin_spatial(img, size=(16, 16)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
