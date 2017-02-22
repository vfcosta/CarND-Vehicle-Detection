##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---
###Project Organization
All source files below have been placed in [`vehicledetection`](vehicledetection) folder:

- [`features.py`](vehicledetection/features.py): Define methods for feature extraction. 
- [`sliding_window.py`](vehicledetection/sliding_window.py): Determine window list to search for vehicles. 
- [`search.py`](vehicledetection/search.py): Iterate through windows and apply feature extraction. 
- [`classifier.py`](vehicledetection/classifier.py): Define Classifier class responsible to train and classify vehicles. 
- [`draw.py`](vehicledetection/draw.py): Helper functions to draw on frame images.
- [`heatmap.py`](vehicledetection/heatmap.py): Define a Heatmap class that receive classifications and generate a heat map of detections. 
- [`pipeline.py`](vehicledetection/pipeline.py): Define an execution pipeline for vehicle detection.
- [`process_samples.py`](vehicledetection/process.py): Execute pipeline steps on sample images.
- [`process_video.py`](vehicledetection/process_video.py): Process video frames using the vehicle detection pipeline.

###Feature Extraction

A module responsible for feature extraction was defined in [`features.py`](vehicledetection/features.py).
The [`single_img_features`](vehicledetection/features.py#L47) function uses three strategies to extract features from input images.
This function also convert to a target color space passed as parameter. 

####1. Histograms of Color

A function [`color_hist`](vehicledetection/features.py#L35) to calculate histograms of color was defined.
It compute the histogram of the color channels separately and return a concatenated array of features.

Example of histogram applied to an input image:

<img src="test_images/25.png" width="180">
<img src="output_images/25_histogram.png" width="250">

####2. Spatial Binning of Color

The function [`bin_spatial`](vehicledetection/features.py#L28) just resize the input image to a fixed size (normally a smaller size) and return a vector of features. 

Example of spatial binning of color:

<img src="test_images/25.png" width="180">
<img src="output_images/25_spatial.png" width="180">

####3. Histogram of Oriented Gradients (HOG)

The function [`get_hog_features`](vehicledetection/features.py#L7) calculates the Histogram of Oriented Gradients (HOG) of an input image.
It uses the opencv scikit-learn `hog` function to calculate the HOG using a set of input parameters.
 
The parameters used in this project was:
    ```
    orient = 12 # HOG  
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # All channels from an image with HLS color space
    ```

See above a visualization of HOG applied to an image:

<img src="test_images/25.png" width="180">
<img src="output_images/25_hog.png" width="180">

###SVM Classifier
Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

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

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

