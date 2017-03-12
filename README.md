# Vehicle-Detection
Detecting Vehicles for Udacity CarND Term 1 Project 5

In this project, I use histograms of oriented gradients (HOGs) and color features, along with a linear support vector machine classifier, in order to detect vehicles in a road video. I use multiple scales of the image classifiers in order to potentially detect vehicles at different distances. To help smooth out multiple detections of the same vehicle and remove false positives, I utilize heat maps to create bounding boxes on the areas with the most detections occuring. 

###Histogram of Oriented Gradients (HOG)

The code for this step is contained in the detect_functions.py file, and the parameters fed into the HOG can be found in the vehicle_classifier.py file, lines 34-45. After training the classifier, I saved the parameters into a pickle file (found at classifier_info.p) to bring into my main pipeline.

I started by reading in all the 'vehicle' and 'non-vehicle' images.  Here is an example of one of each of the 'vehicle' and 'non-vehicle' classes:

![alt text][image1]

I then explored different color spaces and different 'skimage.hog()' parameters ('orientations', 'pixels_per_cell', and 'cells_per_block').  I grabbed random images from each of the two classes and displayed them to get a feel for what the 'skimage.hog()' output looks like.

Here is an example using the 'YCrCb' color space and HOG parameters of 'orientations=9', 'pixels_per_cell=8' and 'cells_per_block=2':

![alt text][image2]

I tried various combinations of parameters, originally staying with RGB for color space, as in the images I looked at originally before the classifier, the color space did not appear to have a big effect. However, I found I gained over an additional percentage of test accuracy (from just over 97% to ~99%) on my SVC classifier by using the 'YCrCb' color space.

I also found that 'orientations' of below 7 did not do a good job at vehicle detection, but those above it did. I settled at 9 orientations based on classifier performance.

For pixels per cell and cells per block, I also tried a few different combinations (less or more), and found 8 pixels per cell and 2 cells per block to have the best performance.

Given that I also used color features, I played around with the number of histogram bins and spatial dimensions. On a small dataset, more histogram bins and less spatial dimensions than my final seemed to perform better, but once I expanded to the full dataset, I found that 16 histogram bins and (16, 16) for spatial dimensions maximized accuracy.

Now, onto the training of a classifier. In lines 47-58 of vehicle_classifier.py, I extract the features from 'vehicle' and 'non-vehicle' using the above parameters, and then use lines 60-69 of the same file to create 'X' from the extracted features, scale it (using StandardScaler from sklearn), and then create labels for 'y' by labelling all vehicles as '1' and all non-vehicles as '0'.

Next, I shuffled and split the data into training and test splits using functions from sklearn. I then used sklearn's LinearSVC (a linear support vector machine classifier) to train my classifier. I saved the classifier, X_scaler, and parameter information to the pickle file 'classifier_info.p' to use in my sliding window search coming up below. This classifier had a test accuracy of 98.99%.

###Sliding Window Search

My sliding window search is implemented in lines 54-111 in 'full_detection_pipeline.py'. For each scale input (from Line 28 in the same file), it will extract the HOG features from each color channel in the 'YCrCb' color space. Based on the window size input, it will slowly "step" across the image (based on pixels in a cell and the number of those cells in a step), with each window being run through the classifier to determine whether it is predicted as a vehicle or non-vehicle. Note that my implementation also puts in the spatial features and color histogram features along with the HOG features (Lines 100 & 101 of full_detection_pipeline.py).

I set the y-start and y-stop to be '400' and 'None', meaning that a little bit over the top half of the image is not searched, but from there it is searched to the bottom of the image. This is because a car would not be expected to be above the horizon line.

I settled on two for cells_per_step (which affects how much the windows overlap), as going higher tended to cause issues later with my heat map, as the windows were less often on top of each other and therefore not causing my heat map to meet the desired threshold.

As far as scales go, I went with scales larger than the training images had been, which were 64 pixels by 64 pixels. Most of the cars near enough to matter are going to be larger than that, so I went with multiple scales from 1.1 on up to 2.3, spread out by 0.4 each to get some variety. This did a good job of having multiple hits on the actual vehicles, while not having very many overlapping false positives.

My final pipeline searched on four scales using the YCrCb 3-channel HOG features, plus spatially binned color and histograms of color in the feature vector, which provided a sufficient result.  Here are some example images, which are prior to using heatmaps:

![alt text][image4]

### Video Implementation

My project video file was created using the 'video_function_with_lanes.py' file. Note that it contains both vehicle detection and lane line detection. The first video, specific to this project, was made by commenting out the lane line detection portion.

Here's a [link to my video result.](./project_videos/project_vid_output.mp4)

This first video is the result of the full detection pipeline, which also utilizes heat maps as I'll describe more below. It does a fairly decent job, with only a few small pop-ups to the left (one of which actually is a car on the other side of the road, so not actually a false positive), and only a few times where it stops detecting one of the vehicles.

In this [second video](./project_videos/project_vid_output_with_lanes.mp4), I also added in my lane line detection from my Advanced Lane Lines [project here](https://github.com/mvirgo/Advanced-Lane-Lines). Note that the 'cam_calibration.py' file helps calibrate the camera for this (using chessboard images from Udacity's Advanced Lane Lines repo as listed in that file), for which I saved the necessary undistortion information to the 'cam_cal_info.p' file in this repository. This gets fed into the 'lane_detect.py' file, which is then pulled into the 'video_function_with_lanes.py' file.

####Heatmaps

The 'heatmap_functions.py' file and Lines 113-129 of 'full_detection_pipeline.py' implement my heat maps.

In order to create the heat maps, for each box detected for an image/frame of video, these these box positions are recorded such that the areas within are given a certain amount of "heat", whereby more "heat" means there were more detections in that spot. After applying a threshold, only certain higher amounts of heat are left to eliminate some of the false positives. I then used 'scipy.ndimage.measurements.label()' to identify individual heat blobs in the heatmap.  From there, the individual heat blobs are assumed to be vehicles, and bounding boxes are made to cover the area of each of these heat blobs.

Here's an example result showing the heatmap from a series of images, the result of 'scipy.ndimage.measurements.label()' and the bounding boxes then overlaid onto the images:

#### Here are six frames and their corresponding heatmaps:

![alt text][image5]

#### Here is the output of 'scipy.ndimage.measurements.label()' on the integrated heatmap from all six frames:
![alt text][image6]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####Issues and Potential Improvements

The first issue I ran into with this project was simply learning the best parameters to use, a common area of focus in machine learning. I initially played around mostly with the spatial dimensions and number of histogram bins, which had appeared to have more of an importance in lessons prior to the project but which I found to not have that great of an impact on the final classifier accuracy.

Along with this, although I was getting up near 97% accuracy with my classifier, I was quickly switching between implementations where I either had way too many false positives, or did the complete opposite and had trouble getting any vehicle detection once including my heat maps. A big discovery was seeing the impact the YCrCb color space had on the classifier - simply changing to that color space (one which admittedly is not very intuitive to me, such as RGB) improved the classifier accuracy to right near 99%. Although that does not sound like a huge difference, it made a clear difference in the output.

From there, implementing the additional scales took some re-tooling of my original pipeline. Once I did that, I was then detecting too many vehicles again, even with the improved classifier. Once I went back and raised my heatmap threshold, that problem was solved.

My model would have some interesting results if it was driving down a road with cars parked on either side or on a more open road (whereas the project video has a middle divider blocking the other side). Especially with a lot of parked cars, there could be a gigantic bounding box all along the side of the image.

My model does still have a few moments where it detects false positives or stops detecting a vehicle. An additional implementation I could use is to average the box boundaries over a few images, so a single frame or two with a false positive would go away, vehicles would consistently stay detected frame-to-frame, and even the box boundaries themselves would be a little more consistent.

As seen above, if I want to detect vehicles and lanes at the same time, I could probably retool my functions to either run more in parallel or otherwise not draw the vehicle detections onto the image until already pulling the lane line detection as well. This would prevent the issue late in the combined video where the lane line detection fails.

Another area of potential vast improvement is with regards to speed. My current implementation is much, much slower than real time (taking nearly 50 minutes to produce the 1.5 minute project video). 

Lastly, I think deep learning would be another approach (especially since the training data is of a sufficient size), or potentially using other classifier algorithms to see how they fare compare to a Linear SVC.

Overall though, I think my implementation for vehicle detection is a good start!
