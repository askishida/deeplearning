# This code is mostly based on
# https://github.com/yukoba/CnnJapaneseCharacter
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

# Due to the latest Keras(2.0) update, following changes are made:
# https://github.com/keras-team/keras/wiki/Keras-2.0-release-notes
import os
import numpy as np
import scipy.misc
from keras import backend as K
from keras import initializers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

num_classes = 75
data_augmentation = True
epochs = 100
batch_size=16
# input image dimensions
img_rows, img_cols = 64, 64
# img_rows, img_cols = 127, 128

earry_stopping=EarlyStopping(monitor='val_loss',patience=10, verbose=1)

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_hiragana_trained_model_b64y_2_1.h5'



ary = np.load("./dataset/etlgtobfutoji64.npz")['arr_0'].reshape([-1, 64, 64]).astype(np.float32) / 15
#X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
X_train = np.zeros([num_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(num_classes * 160):
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')

#labeling
Y_train = np.repeat(np.arange(num_classes), 160)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices. (transform to keras traindata format)
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

model = Sequential()


#def my_init(shape, name=None):
def my_init(mean, stddev, seed):

    #return initialization.normal(shape, scale=0.1, name=name)
    return initializers.TruncatedNormal(mean=mean, stddev=stddev, seed=seed)
my_init = my_init(mean=0.0, stddev=0.05, seed=None)


# Best val_loss: 0.2018 - val_acc: 0.9408 (just tried only once)
# about 7 hours on GPU (NVIDIA GeForce GT 650M)
def m6_1():
 
    model.add(Conv2D(32, kernel_size=(3, 3),kernel_initializer=my_init,input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3),kernel_initializer=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

 
    model.add(Conv2D(64, (3, 3),kernel_initializer=my_init))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),kernel_initializer=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
 
       
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


def classic_neural():
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


m6_1()
# classic_neural()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])




if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset

        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180) 
        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images. We should set False!
        vertical_flip=False, # randomly flip images
        channel_shift_range=100,# change color
        samplewise_center=False)  # Normalization

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    #early stopping
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=X_train.shape[0],
                    epochs=epochs, validation_data=(X_test, Y_test),callbacks=[earry_stopping],workers=4,initial_epoch=0)






# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

'''
We adopted early-stopping, then this program stopped at epochs = 18.


$ python /Users/***/Documents/hiragana.repo/deeplearning/learnb64y_2.py
/Users/***/anaconda3/envs/tensorflow_py35/lib/python3.5/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
from ._conv import register_converters as _register_converters
Using TensorFlow backend.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 62, 62, 32)        320
_________________________________________________________________
activation_1 (Activation)    (None, 62, 62, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 60, 60, 32)        9248
_________________________________________________________________
activation_2 (Activation)    (None, 60, 60, 32)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 30, 30, 32)        0
_________________________________________________________________
dropout_1 (Dropout)          (None, 30, 30, 32)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 64)        18496
_________________________________________________________________
activation_3 (Activation)    (None, 28, 28, 64)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 26, 26, 64)        36928
_________________________________________________________________
activation_4 (Activation)    (None, 26, 26, 64)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 64)        0
_________________________________________________________________
dropout_2 (Dropout)          (None, 13, 13, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 10816)             0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               2769152
_________________________________________________________________
activation_5 (Activation)    (None, 256)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 75)                19275
_________________________________________________________________
activation_6 (Activation)    (None, 75)                0
=================================================================
Total params: 2,853,419
Trainable params: 2,853,419
Non-trainable params: 0
_________________________________________________________________
Using real-time data augmentation.
Epoch 1/100
2018-03-21 08:56:36.465340: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-03-21 08:56:36.465367: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-03-21 08:56:36.465379: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-03-21 08:56:36.538246: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:865] OS X does not support NUMA - returning NUMA node zero
2018-03-21 08:56:36.538927: I tensorflow/core/common_runtime/gpu/gpu_device.cc:887] Found device 0 with properties:
name: GeForce GT 650M
major: 3 minor: 0 memoryClockRate (GHz) 0.9
pciBusID 0000:01:00.0
Total memory: 1023.69MiB
Free memory: 504.59MiB
2018-03-21 08:56:36.538956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:908] DMA: 0
2018-03-21 08:56:36.538965: I tensorflow/core/common_runtime/gpu/gpu_device.cc:918] 0:   Y
2018-03-21 08:56:36.538978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GT 650M, pci bus id: 0000:01:00.0)
9600/9600 [==============================] - 1372s 143ms/step - loss: 4.0280 - acc: 0.0692 - val_loss: 0.5680 - val_acc: 0.8092
Epoch 2/100
9600/9600 [==============================] - 1305s 136ms/step - loss: 3.7852 - acc: 0.1235 - val_loss: 0.3490 - val_acc: 0.8862
Epoch 3/100
9600/9600 [==============================] - 1292s 135ms/step - loss: 3.7411 - acc: 0.1371 - val_loss: 0.3096 - val_acc: 0.8958
Epoch 4/100
9600/9600 [==============================] - 1293s 135ms/step - loss: 3.7268 - acc: 0.1409 - val_loss: 0.2694 - val_acc: 0.9183
Epoch 5/100
9600/9600 [==============================] - 1301s 136ms/step - loss: 3.7140 - acc: 0.1447 - val_loss: 0.2391 - val_acc: 0.9221
Epoch 6/100
9600/9600 [==============================] - 1305s 136ms/step - loss: 3.7156 - acc: 0.1455 - val_loss: 0.2280 - val_acc: 0.9275
Epoch 7/100
9600/9600 [==============================] - 1296s 135ms/step - loss: 3.7106 - acc: 0.1472 - val_loss: 0.2147 - val_acc: 0.9292
Epoch 8/100
9600/9600 [==============================] - 1311s 137ms/step - loss: 3.7092 - acc: 0.1477 - val_loss: 0.1993 - val_acc: 0.9379
Epoch 9/100
9600/9600 [==============================] - 1304s 136ms/step - loss: 3.7090 - acc: 0.1477 - val_loss: 0.2106 - val_acc: 0.9304
Epoch 10/100
9600/9600 [==============================] - 1333s 139ms/step - loss: 3.7056 - acc: 0.1484 - val_loss: 0.2309 - val_acc: 0.9258
Epoch 11/100
9600/9600 [==============================] - 1441s 150ms/step - loss: 3.7186 - acc: 0.1458 - val_loss: 0.2194 - val_acc: 0.9313
Epoch 12/100
9600/9600 [==============================] - 1350s 141ms/step - loss: 3.7165 - acc: 0.1465 - val_loss: 0.2227 - val_acc: 0.9279
Epoch 13/100
9600/9600 [==============================] - 1340s 140ms/step - loss: 3.7093 - acc: 0.1483 - val_loss: 0.2115 - val_acc: 0.9342
Epoch 14/100
9600/9600 [==============================] - 1331s 139ms/step - loss: 3.7052 - acc: 0.1490 - val_loss: 0.2145 - val_acc: 0.9400
Epoch 15/100
9600/9600 [==============================] - 1323s 138ms/step - loss: 3.7213 - acc: 0.1462 - val_loss: 0.2824 - val_acc: 0.9146
Epoch 16/100
9600/9600 [==============================] - 1322s 138ms/step - loss: 3.7236 - acc: 0.1455 - val_loss: 0.3207 - val_acc: 0.9012
Epoch 17/100
9600/9600 [==============================] - 1304s 136ms/step - loss: 3.7265 - acc: 0.1454 - val_loss: 0.2018 - val_acc: 0.9408
Epoch 18/100
9600/9600 [==============================] - 1302s 136ms/step - loss: 3.7287 - acc: 0.1453 - val_loss: 0.2336 - val_acc: 0.9308
Epoch 00018: early stopping
Saved trained model at /Users/kuromame/saved_models/keras_hiragana_trained_model_b64y_2_1.h5
2400/2400 [==============================] - 4s 2ms/step
Test loss: 0.23362154187013706
Test accuracy: 0.9308333333333333

'''
