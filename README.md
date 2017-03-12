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

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
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

Lastly, I think deep learning would be another approach (especially since the training data is of a sufficient size), or potentially using other classifier algorithms to see how they fare compare to a Linear SVC.

Overall though, I think the implementation of vehicle detection is a good start!
