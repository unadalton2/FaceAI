import tensorflow as tf #for ML
import numpy as np      #for data normalization
import cv2              #for img managment
import os               #for file managment

import glob
from PIL import Image

frames = 50

print("Retreaving Data")
#Getting data
imgAPath = "data/imgA.JPG"
imgBPath = "data/imgB.JPG"
height,width = 256, 128 # must be multible of 2


imgA = cv2.imread(imgAPath)#read data
imgB = cv2.imread(imgBPath)#read data

# resize image
imgA = cv2.resize(imgA, (width, height), interpolation = cv2.INTER_AREA)#fit to img size
imgB = cv2.resize(imgB, (width, height), interpolation = cv2.INTER_AREA)#fit to img size

#normalization
imgA = np.array(imgA, dtype=float)/256
imgB = np.array(imgB, dtype=float)/256

print("Creating Model:")
# create model
kernalSize = 8

print("    -Creating Encoder")
#create encoder
imgInput = tf.keras.Input(shape=(256, 128, 3))

x = tf.keras.layers.Conv2D(6, kernalSize, strides=(2,2), padding='same', activation='relu')(imgInput)
x = tf.keras.layers.Conv2D(12, kernalSize, strides=(2,2), padding='same', activation='relu')(x)
encoded = tf.keras.layers.Conv2D(24, kernalSize, strides=(2,2), padding='same', activation='relu')(x)

encoder = tf.keras.Model(imgInput, encoded)

print("    -Creating Decoder")
#create decoder
decoderInput = tf.keras.Input(shape=(32, 16, 24))

x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(decoderInput)
x = tf.keras.layers.Conv2D(12, kernalSize, strides=(1,1), padding='same', activation='relu')(x)

x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
x = tf.keras.layers.Conv2D(6, kernalSize, strides=(1,1), padding='same', activation='relu')(x)

x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
decoded = tf.keras.layers.Conv2D(3, kernalSize, strides=(1,1), padding='same', activation='sigmoid')(x)

decoder = tf.keras.Model(decoderInput, decoded)

print("    -Finalizing Model")
#Create full model
ModelInput = tf.keras.Input(shape=(256, 128, 3))
encoded = encoder(ModelInput)
decoded = decoder(encoded)

model = tf.keras.Model(ModelInput, decoded)

if True:
    model.load_weights("savedModel/")
    
    #Encode imgs
    imgAEncoded = encoder.predict(np.reshape(imgA, (-1, height, width, 3)))
    imgBEncoded = encoder.predict(np.reshape(imgB, (-1, height, width, 3)))

    frameData = []
    for i in range(frames*2):
        #Decode interpolated img
        interpolation = abs((i-frames)/(frames-1))
        interpolatedImg = imgAEncoded * (1-interpolation) + imgBEncoded * interpolation

        outputImg = decoder.predict(interpolatedImg)
        outputImg = cv2.resize(np.reshape(outputImg*256, (height, width, 3)), (150, 210))

        cv2.imwrite("interpolation/interpolation"+str(i)+".jpg", outputImg)
        frameData.append(Image.open("interpolation/interpolation"+str(i)+".jpg"))

    frame_one = frameData[0]
    frame_one.save("interpolation0.gif", format="GIF", append_images=frameData,
               save_all=True, duration=100, loop=0)
    
else:
    print("Unable to load previous model")
