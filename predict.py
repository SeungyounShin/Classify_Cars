from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
import os

path = '/Users/seungyoun/Desktop/CNN_Classification/data/train/2017 kia morning'
#image load&processing
lst = os.listdir(path)
del lst[0]

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def predict(path,any_num):
    print(any_num)
    img = load_img(path)
    #img = load_img('test/grandeur.jpg')
    x = img_to_array(img)  #  (3,img_w,img_h)
    y = tf.image.resize_images(
        x,
        [244,244],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False
        )
    tmp = tf.Session().run(y)
    z = np.reshape(tmp,(1,244,244,3))

    # evaluate loaded model on test data
    predictions = loaded_model.predict(z)
    predict_ = np.reshape(predictions, (4))

    i = np.argmax(predictions)

    print("===========Prediction===========\n",predict_,"\nX :",i)
    return i

count = [0,0,0,0]
j=1
for i in lst:
    go = path + '/' + i
    cnt = predict(go,j)
    count[cnt] += 1
    j += 1
    if(j>10):
        break
print(count)
