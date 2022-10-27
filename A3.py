from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, multiply, Reshape
from pathlib import Path
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
import pandas as pd

vgg16_model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# vgg16_model.summary()

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    # if K.image_data_format() == 'channels_first':
    #     se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def create_model(input_shape, n_classes, optimizer='rmsprop'):
    conv_base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)

    for layer in conv_base.layers:
            layer.trainable = False

    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)
    
    model = Model(inputs=conv_base.input, outputs=output_layer)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def create_SEN_model(input_shape, n_classes, optimizer='rmsprop'):

    base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)

    for layer in base.layers:
        layer.trainable = False

    # Add SEN modules
    SEN = base.layers[0].output
    SEN = squeeze_excite_block(SEN)
    SEN = base.layers[1](SEN)
    SEN = base.layers[2](SEN)
    SEN = base.layers[3](SEN)
    SEN = squeeze_excite_block(SEN)
    SEN = base.layers[4](SEN)
    SEN = base.layers[5](SEN)
    SEN = base.layers[6](SEN)
    SEN = squeeze_excite_block(SEN)
    SEN = base.layers[7](SEN)
    SEN = base.layers[8](SEN)
    SEN = base.layers[9](SEN)
    SEN = base.layers[10](SEN)
    SEN = squeeze_excite_block(SEN)
    SEN = base.layers[11](SEN)
    SEN = base.layers[12](SEN)
    SEN = base.layers[13](SEN)
    SEN = base.layers[14](SEN)
    SEN = squeeze_excite_block(SEN)
    SEN = base.layers[15](SEN)
    SEN = base.layers[16](SEN)
    SEN = base.layers[17](SEN)
    SEN = base.layers[18](SEN)
    SEN = squeeze_excite_block(SEN)

    SEN = Flatten(name="flatten")(SEN)
    SEN = Dense(4096, activation='relu')(SEN)
    SEN = Dense(1072, activation='relu')(SEN)
    SEN = Dropout(0.2)(SEN)
    output_layer = Dense(n_classes, activation='softmax')(SEN)
    
    model = Model(inputs=base.input, outputs=output_layer)

    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def showFeatures(img_path : str, imgName : str, model, index : int):
    imgName += ".jpg"
    #There is an interpolation method to match the source size with the target size
    #image loaded in PIL (Python Imaging Library)
    img = image.load_img(img_path,color_mode='rgb', target_size=(224, 224))

    # Converts a PIL Image to 3D Numy Array
    x = image.img_to_array(img)
    # Adding the fouth dimension, for number of images
    x = np.expand_dims(x, axis=0)

    #mean centering with respect to Image
    model = Model(inputs=model.inputs, outputs=model.layers[index].output)

    x = preprocess_input(x)
    features = model.predict(x)
    square = 8
    ix = 1
    for _ in range (square):
        for _ in range (square):
            ax = plt.subplot(square,square,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(features[0,:,:,ix-1])
            ix += 1
    plt.savefig(imgName)

BATCH_SIZE = 25

train_generator = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

imagenetteDir = Path('imagenette2-320')

trainDir = imagenetteDir/'train'
testDir = imagenetteDir/'val'

class_subset = sorted(os.listdir(imagenetteDir/'train'))

traingen = train_generator.flow_from_directory(trainDir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='training',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=42)

validgen = train_generator.flow_from_directory(trainDir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)

testgen = test_generator.flow_from_directory(testDir,
                                             target_size=(224, 224),
                                             class_mode=None,
                                             classes=class_subset,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)

input_shape = (224, 224, 3)
optim_1 = Adam(learning_rate=0.001)
n_classes=10

n_steps = 20
n_val_steps = 20
n_epochs = 10

# myModel = create_model(input_shape, n_classes, optim_1)
myModel = create_SEN_model(input_shape, n_classes, optim_1)
myModel.summary()

from livelossplot.inputs.keras import PlotLossesCallback

plot_loss_1 = PlotLossesCallback()

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')


history = myModel.fit(traingen,
                            batch_size=BATCH_SIZE,
                            epochs=n_epochs,
                            validation_data=validgen,
                            steps_per_epoch=n_steps,
                            validation_steps=n_val_steps,
                            callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
                            verbose=1)


np.save('history1.npy',history.history)
 
history1=np.load('SEN.npy',allow_pickle='TRUE').item()

pd.DataFrame(history1).plot(figsize=(8,5))
plt.savefig("loss.jpg")

plt.plot(history1['accuracy'])
plt.plot(history1['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig("accuracy.jpg")

plt.plot(history1['loss'])
plt.plot(history1['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig("loss.jpg")

# myModel = create_model(input_shape, n_classes, optim_1)
# myModel = create_SEN_model(input_shape, n_classes, optim_1)

# myModel.load_weights('noSEN.hdf5') 

# for i, layer in enumerate(myModel.layers):
#     print(i, layer.name)

def processImage(path):
    img = image.load_img(img_path,color_mode='rgb', target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

# img_path = '/home/joelsare@unomaha.edu/Documents/A3/Pictures/1.JPEG'
# x = processImage(img_path)
# features = myModel.predict(x)
# print("image 1:")
# print(features)
# print(np.argmax(features))

# img_path = '/home/joelsare@unomaha.edu/Documents/A3/Pictures/2.JPEG'
# x = processImage(img_path)
# features = myModel.predict(x)
# print("image 2:")
# print(features)
# print(np.argmax(features))

# img_path = '/home/joelsare@unomaha.edu/Documents/A3/Pictures/3.JPEG'
# x = processImage(img_path)
# features = myModel.predict(x)
# print("image 3:")
# print(features)
# print(np.argmax(features))

# img_path = '/home/joelsare@unomaha.edu/Documents/A3/Pictures/4.JPEG'
# x = processImage(img_path)
# features = myModel.predict(x)
# print("image 4:")
# print(features)
# print(np.argmax(features))

# showFeatures(img_path, "Pictures/4noSENfeatures18", myModel, 18)