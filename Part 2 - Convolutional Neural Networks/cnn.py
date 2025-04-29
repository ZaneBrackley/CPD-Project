import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Setup the tranformations of the images, so there is no data leakage between this and test set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Load from directory, resize to 64x64, batch size of 32 and 2 outputs.
training_set = train_datagen.flow_from_directory(
            'Part 2 - Convolutional Neural Networks/dataset/training_set',
            target_size = (64, 64),
            batch_size = 32,
            class_mode = 'binary')

# Rescale but no transformations as these are 'new' images
test_datagen = ImageDataGenerator(rescale = 1./255)

# Load from directory with the same filters as before
test_set = test_datagen.flow_from_directory(
            'Part 2 - Convolutional Neural Networks/dataset/test_set',
            target_size = (64, 64),
            batch_size = 32,
            class_mode = 'binary')

# Initialise the CNN
cnn = keras.models.Sequential()

# Convolution layer and pooling layer. Filters is features, kernal size is the grid size of features, relu activation, and image size, with 3 channels for RGB
cnn.add(keras.layers.Conv2D(filters=32, kernal_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Second conv and pool layer, no need for input shape as it has already been connected to the image in first layer
cnn.add(keras.layers.Conv2D(filters=32, kernal_size=3, activation='relu'))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Flatten the image 
cnn.add(keras.layers.Flatten())

# Full connection, dense layer from ANN, with a lot more neurons
cnn.add(keras.layers.Dense(units=128, activation='relu'))

# Binary output layer, same as ANN
cnn.add(keras.layers.Dense(units=1, activation='sigmoid'))