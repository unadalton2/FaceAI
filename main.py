import tensorflow as tf #for ML
import numpy as np      #for data normalization
import cv2              #for img managment
import os               #for file managment

print("Retreaving Data")
#Getting data
path = "data/"
height,width = 256, 128 # must be multible of 2

imgs = []

dirs = os.listdir(path)
for filename in dirs:
    img = cv2.imread(path+filename)#read data
    
    # resize image
    imgs.append(cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA))#fit to img size

#normalization
imgs = np.array(imgs, dtype=float)/256

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

print("    -Loading Weights")
#attempt to load weights
try:
    model.load_weights("savedModel/")
except:
    print("         -Unable to load previous model")

#comple model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy())
#build model
model.fit(imgs, imgs, epochs=0)

#save refrence imgs
cv2.imwrite("OriginalImg.jpg", imgs[0]*256)

img2 = model.predict(np.reshape(imgs[0], (-1, height, width, 3)))
img2 = np.reshape(img2*256, (height, width, 3))
cv2.imwrite("BeforeTraining.jpg", img2)

model.fit(imgs, imgs, epochs=1)

#save test img
img2 = model.predict(np.reshape(imgs[0], (-1, height, width, 3)))
img2 = np.reshape(img2*256, (height, width, 3))
cv2.imwrite("AfterTraining.jpg", img2)

#save weights
model.save_weights("savedModel/", overwrite=True)
