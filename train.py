import csv
import cv2
import numpy as np

train_data_sets = ['train_data', 'train_in', 'train_reverse', 'train_correction']
lines = []

import os
cwd = os.getcwd()

# Go through each directory and construct the list of all training images.
for data_path in train_data_sets:
	csv_path = './' + data_path + '/driving_log.csv'
	print(csv_path)
	with open(csv_path) as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			source_path = line[0]
			filename = source_path.split('/')[-1]
			current_path = cwd + '/' + data_path + '/IMG/' + filename
			line[0] = current_path
			lines.append(line)

from sklearn.model_selection import train_test_split
import sklearn
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                image_flipped = np.fliplr(center_image)
                images.append(image_flipped)
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

print('Training set size : ', len(lines))

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

# LeNet architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*2, validation_data=validation_generator, 
	nb_val_samples=len(validation_samples)*2, nb_epoch=3)

model.save('model.h5') 
