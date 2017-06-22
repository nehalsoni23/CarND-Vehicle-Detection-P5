##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/sample-data.jpg
[image2]: ./output_images/sample-data_2.jpg
[image3]: ./output_images/spatial_binning_1.jpg
[image4]: ./test_images/cutout1.jpg
[image5]: ./output_images/RGB_channel_hist.jpg
[image6]: ./output_images/YCrCb_channel_hist.jpg
[image7]: ./output_images/HOG_vis_car.jpg
[image8]: ./output_images/HOG_vis_non_car.jpg
[image9]: ./output_images/only_one_window.jpg
[image10]: ./output_images/finding_cars_1.jpg
[image11]: ./output_images/cropped_image.jpg
[image12]: ./output_images/finding_cars.jpg
[image13]: ./output_images/finding_cars_2.jpg
[image14]: ./output_images/heatmap_1.jpg
[image15]: ./output_images/heatmap_2.jpg
[image16]: ./output_images/output_image.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 5th code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]	![alt text][image2]

I applied spatial binning on an image. Using full resolution images to extract feautures would be time taken when with lower resolution also we can retain enough information to help finding vehicles. See the example below where 32x32 resolution is applied with the use of `cv2.resize()`.

![alt text][image3]

Then I obtain color histograms of three color channel on an image with 32 bins defined and concatenated three histograms into single array. See below image showing individual color channel histograms for RGB and YCrCb colorspace.

![alt text][image4]
![alt text][image5] ![alt text][image6]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image7]

![alt text][image8]

#### 2. Explain how you settled on your final choice of HOG parameters.

After trying various parameters I kept HOG parameters as given in classroom lessons which are pix_per_cell:  8
cell_per_block:  2, orient:  9.

Although while sliding window in `find_cars` function, I had to change one parameter naming `cells_per_step = 1` which defines overlap in terms of steps. Earlier it was 2, but becasue of this less overlapping when my car is bit far from screen it was detecting very less frame. See below image for that.

![alt text][image9]

And while applying threshold on heatmap it would remove this single frame also resulting no detection of a car for few seconds.

After changing `cells_per_step = 1` with broadening my ystart a bit I was able to detect car which are bit far on horizon as shown below. Although then I changed my threshold value to 2 to avoid more false detections.

![alt text][image10]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My SVM classifier is written in cell 10. First I extracted the features of all the car and non-car images. I stacked those features and to constitute training data. Then I normalize the data using sklearn's `StandardScaler()` method. Then I created labels for these training data by assigning 1 to car and 0 to non-car images.

After creating feature dataset from images, I splitted the data into training and test dataset using `train_test_split` method of sklearn. I kept 20% of total data for testing. Before feeding data to SVM, I shuffled data in order to avoid data ordering and overfitting. I used LinearSVC from `sklearn.svm` package. Once model is trained, it was tested on test dataset and accuracy obtained is 0.9916.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

Before sliding a window through the image, I cropped an image as shown below by defining ystart and ystop value in order to prevent it searching cars in sky and treetops.

![alt text][image11]

As per `pix_per_cell` and `cell_per_block`, total blocks are calculated in an image. I have used hog subsampling for window search as shown in [this classroom lesson](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/c3e815c7-1794-4854-8842-5d7b96276642). In which I am extracting hog feature for an entire image first instead of finding HOG feature for each window in an single image. And then from every window subsampling features from an image hog feature. 

Then as explained in step 1 of `Histogram of Oriented Gradients (HOG)` topic, I am combining features of spatial binning, color histograms and hog to predict if it is car or non-car. If the window contains car then I am storing the result in an array in order to draw an rectangle on an image using `cv2.rectangle()` function.


#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image12] ![alt text][image13]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./processed_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three frames and their corresponding heatmaps:

![alt text][image14] ![alt text][image15]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap along with the resulting bounding boxes drawn on a single frame:

![alt text][image16]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The main useful approach I used is HOG subsampling because of which the video processing time is much reduced and pipeline is efficient. 
The problem is faced during working on project is no detection of white car when it is bit far in an image as I explained in `Histogram of 
Oriented Gradients (HOG)` step 2. To overcome this problem, I changed overlapping parameter defined in step format `cells_per_step = 1`. Due to which the number of detection has been increased and then I increased threshold to 2 in order to prevent that. If I am extending this project further, then I will reduce the video processing time by choosing the some of frames to process from raw video. Also I will add traffic sign detection, pedestrian detection and the must needed LAN detection which I leanrned in 4th module.