# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model


def Cnn_layers(training_set, test_set):
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    classifier.fit_generator(training_set,
                             steps_per_epoch = 120,
                             epochs = 7,
                             validation_data = test_set,
                             validation_steps = 17)
    classifier.save('CrackDetectionGauravModel2.h5')
    return classifier


# Part 2 - Fitting the CNN to the images
def input_data():
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('C:/Users/A638054/PycharmProjects/SurfaceFaultDetectionCnn/dataset1/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
    test_set = test_datagen.flow_from_directory('C:/Users/A638054/PycharmProjects/SurfaceFaultDetectionCnn/dataset1/test_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
    return training_set, test_set


def load_saved_model():
    return load_model('CrackDetectionGauravModel2.h5')



# Part 3 - Making new predictions
def Predictions():
    model = load_saved_model()
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('image1.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    return result






training_set, test_set = input_data()
classifier = Cnn_layers(training_set,test_set)
result = Predictions()


    #training_set.class_indices
if result[0][0] == 1:
    prediction = 'NoDefect'
    print('Nodefect')
else:
    prediction = 'Defect'
    print('Defect')
