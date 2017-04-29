
**Vehicle Detection Project**
---

[windows]: ./output_images/windows.jpg
[weightmap]: ./output_images/weightmap.jpg
[heatmap]: ./output_images/heatmap.jpg
[normalized]: ./output_images/heatmap_norm.jpg
[boxes]: ./output_images/boxes.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step resides in [vehicle/features.py](./vehicle/features.py).

I have selected HSV colorspace and used HOG only on saturation and value channels. From my previous experience 
hue channel does not usually contain useful information. 
I have also used other features, provided by the instructor.

Here is the list of features:
* HOG with 9 orientations, 16 pixels per cell and 2 cells per block. (For saturation and value)
* Spatial binning (basically downscaled image itself) downscaled to (16 x 16)
* Color histograms for each of three channels (32 bins each) 


#### 2. Explain how you settled on your final choice of HOG parameters.

I have tried various parameters, it seems that the only thing they affect is the computation speed, as the classifier almost always 
produces accuracy around 97% - 99% on the test data. Most important was to have a high number of search windows.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier and scaler is in [train.py](./train.py).

I have used linear SVM and Standard scaler.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried to use the knowledge from the previous lane finding problem and define search windows in real world coordinates,
which could be potentially used later for estimating distance and direction to the detected vehicle.
The code for it can be found in [windows.py](./windows.py)

Windows used.
![windows]

Each window has been scaled down to have the height of 128, in which the search was made with patches of (64, 64) and
steps of (16, 16).

Initially, I had a large number of windows each was located at an interval of 1 meter, but this was too heavy. 
So I decided to go with scaling each interval. Scale was set to 1.15.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

One frame processing takes about 3.5 seconds, which is quite long. Unfortunately, I didn't have much time to optimize it.

Detected boxes.
![boxes]

Heatmap.
![heatmap]

Normalized heatmap.
![normalized]

Weight map. This is the weightmap which I used to normalize heatmap. It is computed along with the windows and basically
each pixel has a number of patches it is present in.
![weightmap]

Also before labeling I have used morphological opening procedure.

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I didn't do any frame-to-frame tracking.
As for the false positives, I have collected around 800 negatives using hard negative mining. For this I have created a script which was presenting a patch
and then it got stored to negative folder (see [mining](./vehicle/pipeline.py)). Also, pretty limited search windows also prevent 
from getting false positives. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First of all, I am upset with the performance. I didn't implement HOG features reuse between patches, 
lack of time is to blame. 

Pipeline will likely have some false positives, from time to time. Also, I am not quite happy with 
the bounding box combination algorithm, this could be more robust and accurate.

