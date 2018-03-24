

# Applying handwriting-letter dataset to identifying printing-type one in Japanese Hiragana CNN-train model
------------------

Using handwriting Hiragana ETS8G-based dataset both to train identifying model and to predict for bold-face Hiragana character as like signboard by keras.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

<img src="https://github.com/askishida/deeplearning/blob/master/sample_images/pi2.jpg" width="300">


------------------

Overview

## Description

This source cord is mostly based on following three works and special thanks to: 


https://github.com/yukoba/CnnJapaneseCharacter
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py



 At first this source-code is able to extract only Hiragana-letters(127 x 128 px) 
from ETS8G dataset trim margin, thicken, and finished to resized binary-dataset
(64 x 64 px).

The original color ETS8G dataset is provided by AIST in Japan and each one is 
written by 160 people in Japan. 

http://www.aist.go.jp/index_en.html

 In contrast, this arranged-dataset consists of 75 Hiragana character in the following order.
 

'あいうえおかがきゃぎ　くぐけげこごさざしゅ　じすずせぜそぞただち　ょぢつづてでとどなに　
っぬねのはばぱひびぴ　ふぶぷへべぺほぼぽま　みむめもやゆよらりる　れろわをん'



 We can easily find why we need this preprocess, Because Hiragana-letter identifying is 
so delicate both to margin and thickness of letter. 

 At next source-code, we can train CNN-model by using previous arranged dataset.

 At third source-code, we read some preprocessed imagefile which contains Hiragana-letter 
on car-number plateboard and over again make thin this letter close to same thickness of dataset.
And we can predict this by using previous train model.

We found that this model could identify as like difference between "ば" and "ぱ"
but couldn't do small-Hiragana as like one between "ゆ", "ゅ" now.

 We should do another approach as like pre-detecting text-area to which we need to evaluate
the ratio of margin-area in addition.
And we should need to output identification rate of each character and desgin another algorithm 
for these low-score characters.

## Demo
my arranged Hiragana dataset here:

dataset      ----- gtobfutoji64yohaku.npz

my trained model here:

saved_models ----- keras_hiragana_trained_model_b2.h5

## Requirement

anaconda3-python3.5
scikit-learn (0.19.*)
numpy (1.14.*)
scipy (1.0.*)
matplotlib (2.1.*)
Pillow (4.2.*)
keras (>= 2.0)
Tensorflow
h5py(2.7.*)
HDF5(1.10.*)


## Usage

in case Linux or MacOSX
-in your tarminal
```sh
cd your filepath-derectory
python3 source-code-name.py  
```
in case Windows10 OS
-in your ComandPronpt
```sh
cd your filepath-derectory
python3 source-code-name.py
```
## Install
 At first, you get anaconda3 installers and open Anaconda-Navigater and 
create a virtual environment with python3.5.
 Next, you open terminal of this virtual environment and
input as follow.

```sh
pip install numpy, pillow, scipy, matplotlib, keras, tensorflow, h5py, hdf5
```

## Contribution

 I mainly report to upgrade previous works by author .. written in keras1.0 code to keras2.0 
and eliminate some bugs in arranged-dataset and improve it up.
 
 Original ETS8G dataset consists of both Chinese character and Japanese character 
and so author Mr.yukoba's sourcecode is written in Keras1.0 and tried to extract only 
Hiragana-letters(127 x 128 pixel) but by misfortune, his dataset contains some Chinese characters.
In contrast, his dataset have 72 characters, mine 75 one. 
But as considering the conclusion, we should adopt 71 characters containing one as like "ば", "ぱ" and exclude small characters as like "ゃ","ゅ", "ょ", "っ". 

 He adopt 32 x 32 pixel, but I am afraid that that size is too small for us to identify 
as like difference between "ば"  and "ぱ". 
 Decisively different from his work, in this work we are concerned with applying it to identifying 
other kind of letters from handwriting one, although we use handwriting source as a raw material.
As a result, We can easily find that either thickness or margin in letter-image seriously affect on 
the recognition rate and so my train-model score is lower than him.   

 Specifically in next my work, I tried to arrange original hand-writing-dataset to apply to identifying 
a letter on signboard. In this proccess, as a result of trial and error, I found that doing Hiragana 
learning need 64 x 64 pixel size dataset.
And so carefully did I trim margin of image files. Indeed, difference between "あ" and "わ" depends
on margin, which mean feature of Japanese character and so hard is it to distinguish.
There are similar combinations exist further more as described above.
Although I added images processing apparatus to this pre-arranged dataset by keras's fit_generator, 
we couldn't easily find that amount of data in original dataset was too small to train enough.
In other words, we should collect more and more sample-images to make dataset, and devise more 
appropriate methods for padding data algorithm.

 At finaly I brought in a new jpeg picture-file of a letter on number-plate of car and predicted it 
by previous train model. In addition to it, I output proposed list set with probability in more 
likely order.



## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[askishida](https://github.com/askishida)









