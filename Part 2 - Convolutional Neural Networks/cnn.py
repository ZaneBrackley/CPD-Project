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

# Initialise the CNN, needs to be done like this as input_shape is not a valid parameter for Conv2D
cnn = keras.Sequential([
    keras.Input(shape=(64, 64, 3)),
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid'),
])

# Compile the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the training dataset and evaluate on the test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)