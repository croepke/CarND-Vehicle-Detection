import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from helpers import single_img_features

vehicles = glob.glob('images/vehicles/*/*.png')
non_vehicles = glob.glob('images/non-vehicles/*/*.png')

X_vehicle = []
X_non_vehicle = []

color_space = 'BGR' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

for vehicle_image in vehicles:
    veh_img = cv2.imread(vehicle_image)
    features = single_img_features(veh_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    X_vehicle.append(np.ravel(features))

for non_veh_image in non_vehicles:
    non_veh_img = cv2.imread(non_veh_image)
    features = single_img_features(non_veh_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    X_non_vehicle.append(np.ravel(features))

vehicles_y = np.ones(len(vehicles))
non_vehicles_y = np.zeros(len(non_vehicles))

X = X_vehicle + X_non_vehicle
X = np.array(X)
y = np.append(vehicles_y, non_vehicles_y)

X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)

out = {'clf': clf, 'scaler': X_scaler, 'accuracy': acc}

with open('classifier.pickle', 'wb') as handle:
	pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)