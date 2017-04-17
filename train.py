import csv
import cv2
import numpy as np

train_data_sets = ['train_data', 'train_in', 'train_correction']#'train_data_DK', 'train_data_2', 'train_T2']
images = []
measurements = []

for data_path in train_data_sets:
	lines_set = []
	images_set = []
	measurements_set = []

	csv_path = './' + data_path + '/driving_log.csv'
	print(csv_path)
	with open(csv_path) as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			lines_set.append(line)

	for line in lines_set:
		source_path = line[0]
		filename = source_path.split('/')[-1]
		current_path = './' + data_path + '/IMG/' + filename
		#print(current_path)

		image = cv2.imread(current_path)
		images_set.append(image)
		measurement = float(line[3])
		measurements_set.append(measurement)

		image_flipped = np.fliplr(image)
		measurement_flipped = -measurement
		images_set.append(image_flipped)
		measurements_set.append(measurement_flipped)

	images.extend(images_set)
	measurements.extend(measurements_set)

print('Training set size : ', len(images))

X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

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
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5') 
