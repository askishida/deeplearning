#https://github.com/yukoba/CnnJapaneseCharacter/blob/master/src/read_hiragana_file.py
#https://qiita.com/haruhiko28/items/754e355e6e8d8d475cff
import struct
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
sz_record = 8199
#This arranged Hiragana dataset consists of 75 different characters.
num_classes=75




def read_record_ETL8G(f):
    s = f.read(sz_record)
    
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)


def read_hiragana():
    # Character type = 75, person = 160, y = 127, x = 128
    ary = np.zeros([75, 160, 127, 128], dtype=np.uint8)

    for j in range(1, 33):
        filename = '/path/to/ETL8G/ETL8G_{:02d}'.format(j)
        with open(filename, 'rb') as f:
            for id_dataset in range(5):
                moji = 0
                for i in range(956):
                    r = read_record_ETL8G(f)
                    #print(r[2])
                    if not b'HEI.HIRA' in r[2] and not b'KAI.HIRA' in r[2] and not b'SA.SHIRA' in r[2]:
                      if b'.HIRA' in r[2] or b'.SHIRA' in r[2] or b'TSU.SHIR' in r[2] or b'O.WO.HIR' in r[2]:
                          ary[moji, (j - 1) * 5 + id_dataset] = np.array(r[-1])
                          #print((j - 1) * 5 + id_dataset)
                          moji += 1

    ary = ary.reshape([-1, 127, 128]).astype(np.float32) / 15

    for i in range(num_classes * 160):
      ary[i]= scipy.misc.imresize(ary[i], (127, 128), mode='L')
      ary[i][ary[i]<110]=0
      ary[i][ary[i]>=110]=255

      #This here, we apply to it a process of dilation for making letter thick, because of 
      #black-and-white inverted image.  

      ary[i]=ndimage.binary_dilation(ary[i])
      ary[i]=ndimage.binary_dilation(ary[i])
    ary3=np.zeros([num_classes * 160, 97, 97], dtype=np.float32)
    ary2=np.zeros([num_classes * 160, 64, 64], dtype=np.float32)
    for i in range(num_classes * 160):
      #trim margin and resize to 64 x 64 pixel
      lx, ly = ary[i].shape
      # Cropping
      ary3[i] = ary[i][15:112,15:112]    
      ary2[i] = scipy.misc.imresize(ary3[i], (64, 64), mode='L')    
      ary2[i][ary2[i]<110]=0
      ary2[i][ary2[i]>=110]=255

    np.savez_compressed("./deeplearning/dataset/etlgtobfutoji64.npz", ary2)
    plt.imshow(ary2[160*72+1])
read_hiragana()


    


#These characters are arranged as 160 units of each one in the following order.


#plt.imshow(ary2[160*72+1])
#わ
plt.show()





#kana = 'あいうえおかがきゃぎくぐけげこごさざしゅじすずせぜそぞただちょぢつづてでとどなにっぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめ　もやゆよらりるれろわをん'



 


