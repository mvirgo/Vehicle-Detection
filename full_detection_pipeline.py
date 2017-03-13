import numpy as np
import pickle
from scipy.ndimage.measurements import label
from detect_functions import *
from heatmap_functions import *

# Pull in classifier info dictionary from pickle file
classifier_info = pickle.load(open("classifier_info.p", "rb"))

# Pull in parameters and classifier from dictionary
color_space = classifier_info['color_space']
orient = classifier_info['orient']
pix_per_cell = classifier_info['pix_per_cell']
cell_per_block = classifier_info['cell_per_block']
hog_channel = classifier_info['hog_channel']
spatial_size = classifier_info['spatial_size']
hist_bins = classifier_info['hist_bins']
spatial_feat = classifier_info['spatial_feat']
hist_feat = classifier_info['hist_feat']
hog_feat = classifier_info['hog_feat']
y_start_stop = classifier_info['y_start_stop']
ystart = y_start_stop[0]
ystop = y_start_stop[1]
svc = classifier_info['svc']
X_scaler = classifier_info['X_scaler']

# Scales to iterate through for window searches
scale = [1.1, 1.5, 1.9, 2.3]

# Frames to smooth heatmap over
heat_frames = 5

# The higher this is, the further from the decision boundary it is.
# Numbers close to zero are not very confident - this is not percentage confidence
confidence_threshold = 0.3

# Threshold for strength of heat needed to make it to final bounding box
# Helps remove false positives
heat_threshold = 2

# Class to average heatmaps over
class Recent_Heat:
    def __init__(self):
        self.heat_list = []
    def add_heat(self, heat):
        self.heat_list.append(heat)

rec_heat = Recent_Heat()

def convert_color(img, conv='RGB2YCrCb'):
    """ Adding in a few color conversion possibilities.
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img):
    """ This function is the main pipeline for vehicle detection.
    It extracts features using hog sub-sampling, and also uses color features
    in making its predictions using the already trained classifier.
    """
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    # List to put in the box coordinates for drawing
    boxes = []

    # Iterate through for each scale from the list above
    for num in scale:
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        
        if num != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/num), np.int(imshape[0]/num)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # How many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))       
                confidence = svc.decision_function(test_features)
                test_prediction = svc.predict(test_features)

                if test_prediction == 1 and abs(confidence) > confidence_threshold:
                    xbox_left = np.int(xleft*num)
                    ytop_draw = np.int(ytop*num)
                    win_draw = np.int(window*num)
                    boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,boxes)

    # Append new heatmap to recents list
    rec_heat.add_heat(heat)

    # Drop oldest frame to have ten total for average
    if len(rec_heat.heat_list) > heat_frames:
        rec_heat.heat_list = rec_heat.heat_list[1:heat_frames+1]

    # Make into array so np.mean will calculate for each value in the image
    recent_heat_array = np.array(rec_heat.heat_list)
        
    # Take the average heat
    avg_heat = np.mean(np.array(recent_heat_array), axis=0)
    
    # Apply threshold to help remove false positives
    avg_heat = apply_threshold(avg_heat,heat_threshold)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(avg_heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(draw_img, labels)
    
    return draw_img
