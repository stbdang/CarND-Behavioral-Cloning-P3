import csv
import cv2
import numpy as np

lines = []
with open('./train_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		lines.append(line)

images = []
measurements = []
correction = 0.2

for line in lines:
	for i in range(0,3):
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = './train_data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		if i == 1:
			measurement -= correction
		elif i == 2:
			measurement += correction

		measurements.append(measurement)

		image_flipped = np.fliplr(image)
		measurement_flipped = -measurement
		images.append(image_flipped)
		measurements.append(measurement_flipped)

X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5') 
