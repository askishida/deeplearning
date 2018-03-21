# This code is mostly based on
# https://github.com/yukoba/CnnJapaneseCharacter
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

import os
import operator
from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from scipy import ndimage
from keras import backend as K

from keras import initializers

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix


from keras.models import load_model
from scipy import misc

from keras.preprocessing import image

import matplotlib.pyplot as plt





kana=['あ','い','う','え','お','か','が','き','ゃ','ぎ','く','ぐ','け','げ','こ','ご','さ','ざ','し','ゅ','じ','す','ず','せ','ぜ','そ','ぞ','た','だ','ち','ょ','ぢ','つ','づ','て','で','と','ど','な','に','っ','ぬ','ね','の','は','ば','ぱ','ひ','び','ぴ','ふ','ぶ','ぷ','へ','べ','ぺ','ほ','ぼ','ぽ','ま','み','む','め','も','や','ゆ','よ','ら','り','る','れ','ろ','わ','を','ん']
kana_num = len(kana)
print(kana_num)
labels = np.empty([0, kana_num], np.int)

num_classes = 75
data_augmentation = True
epochs = 12
batch_size=16
# input image dimensions
img_rows, img_cols = 64, 64
#img_rows, img_cols = 127, 128




l_model = load_model('./saved_models/keras_hiragana_trained_model_b64y_2_1.h5')


ary = np.load("./dataset/etlgtobfutoji64.npz")['arr_0'].reshape([-1, 64, 64]).astype(np.float32) / 15
X_train = np.zeros([num_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(num_classes * 160):
    
    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
plt.imshow(ary[160*1+150])
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


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)


print([X_test[0]])
print([X_test.shape[0]])#2400
print([Y_test][0])
print([Y_test.shape[0]])#2400







# load a input image. your any sized images are available.
# Your image will be resized to 64 X 64 pxcel size.    
images = np.empty([0, 64, 64], np.float32)
img_ori = Image.open('./sample_images/a.jpg')


resize_img = img_ori.resize((64, 64),Image.LANCZOS)
#img_gray = ImageOps.grayscale(resize_img)
img_gray=resize_img.convert("L")
img_b=img_gray.point(lambda x: 0 if x < 192 else x)
img_b=img_gray.point(lambda x: 255 if x >= 192 else x)
#img_b.show()
img_ary = np.asarray(img_b)

img_ary = 255 - img_ary
img_ary[img_ary<110]=0
img_ary[img_ary>=110]=255

#This here, we apply to it a process of erosion for making letter thin ,because of 
# black-and-white inverted image.  


img_ary=ndimage.binary_erosion(img_ary)
images = np.append(images, [img_ary], axis=0)
#plt.imshow(img_ary)
plt.show()

images = images.reshape(1, 64, 64, 1)

ret = l_model.predict(images, batch_size=16, verbose=0)
print(ret)

index=np.argmax(ret, axis = None, out = None)


for i in range(len(ret)):
    #print(np.argsort(ret)[:, ::-1][i],np.sort(ret)[:, ::-1][i])
    ind=np.argsort(ret)[:, ::-1][i]
    sco=np.sort(ret)[:, ::-1][i]


for i in range(len(ind)):
    print(i+1,'番目の候補は「',kana[ind[i]],'」で',round(sco[i]*100,2),'%です。')
    if i == 3:
      break

print('予想は：「'+kana[index]+'」')
print(ret.shape)



'''
img_ori = Image.open('./sample_images/n.jpg')
[[7.4717915e-10 3.0572902e-13 9.5670396e-09 4.6852356e-06 4.0592307e-09
  3.4400159e-12 1.0795461e-12 1.9476595e-12 2.3112019e-11 5.3702984e-13
  3.3396439e-09 1.0185061e-10 1.4667200e-12 2.8990043e-13 4.0824892e-11
  2.7201231e-13 1.3443004e-13 8.8777227e-15 2.6796506e-11 5.7414420e-11
  5.6479826e-13 6.8599403e-14 1.6381483e-13 2.3106454e-12 3.0792006e-13
  1.5285288e-08 3.1294742e-10 4.2540094e-07 9.6532631e-09 1.1213407e-09
  4.7748388e-13 3.5880812e-11 2.7979659e-09 4.7294313e-12 2.6480276e-11
  3.2719847e-12 1.7221468e-09 3.2122152e-11 2.3310989e-10 1.5542691e-10
  7.3313000e-10 1.0799966e-09 1.9684281e-08 2.9459564e-08 4.1954179e-12
  1.7716873e-12 8.9010234e-14 9.4254541e-13 2.7028376e-14 2.0679156e-14
  6.4556482e-10 2.1624449e-11 5.0333782e-12 5.2097704e-09 3.9559182e-11
  4.3230360e-11 3.6555016e-11 4.6025653e-12 2.5244618e-12 6.3378068e-13
  1.4516147e-10 1.0546349e-12 1.5019964e-10 3.4008612e-11 3.0643689e-11
  1.4264080e-10 1.0091965e-11 3.8387356e-08 4.5142175e-11 2.1981455e-07
  2.5998854e-09 1.0878585e-06 1.5899467e-07 4.8950388e-10 9.9999332e-01]]
1 番目の候補は「 ん 」で 100.0 %です。
2 番目の候補は「 え 」で 0.0 %です。
3 番目の候補は「 ろ 」で 0.0 %です。
4 番目の候補は「 た 」で 0.0 %です。
予想は：「ん」
(1, 75)
正解は:「ん」
img_ori = Image.open('./sample_images/wa.jpg')
[[7.18034653e-06 9.88319059e-15 2.84899464e-14 9.46585317e-08
  2.66131701e-05 1.28827757e-08 2.17972407e-12 1.50621497e-12
  2.17266982e-10 7.18842981e-14 7.07748504e-17 1.05634597e-16
  1.57523801e-14 3.28239799e-16 6.31288785e-15 2.23027947e-15
  1.92500920e-12 1.08119131e-14 2.91515203e-16 3.19567116e-12
  2.29484228e-16 9.39288664e-12 5.00058220e-13 5.87015192e-09
  8.14749310e-12 6.56906585e-10 1.59553704e-10 1.51851310e-07
  3.55360019e-10 1.16106302e-09 4.98125507e-14 4.11233547e-11
  6.23219643e-10 1.94455915e-13 2.98169432e-12 1.37134526e-13
  1.07109495e-10 1.43763040e-11 2.63604338e-08 6.03463541e-14
  1.15366131e-13 4.73681865e-07 1.97656336e-03 1.68152992e-11
  1.31462647e-12 6.83167173e-14 2.68596477e-15 3.01011820e-11
  8.42238317e-15 1.27165897e-13 6.07580349e-12 6.97026504e-15
  8.21217335e-13 2.67157686e-15 7.84528110e-16 1.36371237e-14
  1.33565479e-12 4.17868077e-15 1.12914371e-14 9.65058450e-11
  1.89173677e-09 1.23902555e-05 1.99781496e-07 8.09042522e-09
  2.69255906e-09 4.74815520e-10 3.75829090e-11 5.95071350e-15
  4.13972742e-16 4.73975859e-10 4.53785651e-06 1.38491593e-10
  9.97971714e-01 1.11883098e-08 2.58088839e-09]]
1 番目の候補は「 わ 」で 99.8 %です。
2 番目の候補は「 ね 」で 0.2 %です。
3 番目の候補は「 お 」で 0.0 %です。
4 番目の候補は「 む 」で 0.0 %です。
予想は：「わ」

img_ori = Image.open('./sample_images/pi.jpg')
[[3.13756612e-19 6.30117934e-17 5.71215114e-18 5.98743822e-21
  6.43110655e-15 1.81567576e-12 3.82212040e-10 1.74159344e-21
  2.18585506e-22 2.32542020e-14 1.39904945e-17 7.48550256e-14
  2.26028815e-15 3.36683875e-10 1.15055343e-19 9.36939935e-12
  2.42897483e-20 2.49062771e-14 1.94132110e-17 5.23717975e-21
  4.86806992e-13 5.03245186e-15 2.17446512e-11 1.12680518e-15
  6.46390146e-12 3.12399906e-19 2.69844046e-14 3.61153673e-19
  7.88722793e-14 2.60183094e-17 3.17358208e-19 2.92811747e-11
  2.65138425e-19 6.23125787e-11 2.38721000e-17 9.28514383e-12
  7.36774900e-18 2.51267475e-12 1.25300801e-12 6.07762489e-20
  2.41752016e-21 1.29112584e-21 5.45029427e-21 9.58498354e-16
  3.55475058e-16 2.41988395e-12 5.67791725e-09 1.22838149e-08
  1.75558871e-05 9.99982476e-01 1.32083936e-17 1.81923148e-13
  5.82783999e-09 4.12524608e-18 1.93633807e-15 7.63057965e-14
  5.45560123e-20 4.45512013e-16 1.15791994e-11 1.73025085e-20
  2.63552065e-13 5.58297332e-13 1.48974479e-19 5.77897379e-19
  1.17751052e-18 9.90736745e-16 6.57549198e-20 1.02412884e-17
  8.67470926e-18 1.30991647e-16 1.05573920e-18 3.59205795e-17
  1.58090177e-20 2.23622778e-18 3.05610485e-19]]
1 番目の候補は「 ぴ 」で 100.0 %です。
2 番目の候補は「 び 」で 0.0 %です。
3 番目の候補は「 ひ 」で 0.0 %です。
4 番目の候補は「 ぷ 」で 0.0 %です。
予想は：「ぴ」

'''
