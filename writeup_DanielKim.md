#** Behavioral Cloning** 

## Daniel Kim

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/center.jpg "Center training"
[image2]: ./writeup/left.jpg "Left"
[image3]: ./writeup/right.jpg "Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_DanielKim.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model borrows the structure of a LeNet architecture. It consists of
- Cropped to the region of interest
- Lambda layer (normalization)
- 5x5x6 Convolution with Relu
- Max pool 2x2
- 5x5x6 Convolution with Relu
- Max pool 2x2
- Fully connected with 120 nodes
- Fully connected with 84 nodes

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 5-23). The training data generated for different scenarios were saved to different folders for maintainability.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left, right lane driving and reverse driving. I also added a correction training set to recover from going out of track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to keep the architecture simple to limit the training time and overcome it by providing more training data.

My first step was to create a model of LeNet. I thought this model might be appropriate because LeNet was proven to be pretty effective in image recognition.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I created more test data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I specifically created training data for those handling these scenarios.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-83) is of a LeNet architecture described above.

#### 3. Creation of the Training Set & Training Process

I started off with the sample data set provided. Once I have set up the basic LeNet architecture, I added various pieces which can improve the performance. (e.g. cropping, normalization, adding flipped images to the data...etc) I was able to get the car going in a straight line with the given traing set.

To capture good driving behavior, I then recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle driving closer to the left of the lane and the right of the lane to provide it some boundary data.

![alt text][image2]
![alt text][image3]

Then I recorded a lap of reverse driving to give it a "fresh" data.

After this step, the car was able to handle most of the track except some tight corners or around the object that can confuse the network.

I started recording for handling these cases by recording the the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself.

After the collection process, I had 34312 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the fact that validation error did not change much after the third epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
