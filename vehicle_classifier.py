import matplotlib.image as mpimg
import numpy as np
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from sklearn.utils import shuffle
from detect_functions import *

def vehicle_classifier():
    """ This loads in training images, allows for parameters to be set,
        extracts features from the training images based on those parameters,
        sets X and y and then splits into training and testing sets,
        and then fits a Linear SVC classifier. The trained classifier and
        parameters are saved to a pickle file for future use.
    """

    # Load in the training images
    vehicle_images_loc = glob.glob('vehicles/*/*.png')
    other_images_loc = glob.glob('non-vehicles/*/*.png')

    cars = []
    notcars = []

    for img in vehicle_images_loc:
        cars.append(img)
                
    for img in other_images_loc:
        notcars.append(img)

    ### These parameters can be tweaked to see how the results change.
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [400, None] # Min and max in y to search in slide_window()

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Shuffle the data
    scaled_X, y = shuffle(scaled_X, y)

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    # Print some useful information about the inputs to the classifier
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC 
    svc = LinearSVC()
    
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    # Check the prediction time for a single sample
    t=time.time()

    # Set a dictionary to hold the important information
    classifier_info = {'svc': svc, 'X_scaler': X_scaler, 'color_space': color_space, 'orient': orient,
                       'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block, 'hog_channel': hog_channel,
                       'spatial_size': spatial_size, 'hist_bins': hist_bins, 'spatial_feat': spatial_feat,
                       'hist_feat': hist_feat, 'hog_feat': hog_feat, 'y_start_stop': y_start_stop}

    # Dump the dictionary to a pickle file for future use
    pickle.dump(classifier_info,open('classifier_info.p', "wb"))

    print("Vehicle Detection classifier information saved.")


vehicle_classifier()
