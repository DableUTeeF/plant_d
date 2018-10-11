from __future__ import print_function
from keras.metrics import top_k_categorical_accuracy
from keras import applications as ka
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import json
import sys


def getmodel():
    m = ka.inception_resnet_v2.InceptionResNetV2(classes=61,
                                                 weights=None,
                                                 pooling='avg')
    return m


def top_3_categorical_accuracy(y_true, y_predict):
    return top_k_categorical_accuracy(y_true, y_predict, 3)


if __name__ == '__main__':
    lr = 0.01
    model = getmodel()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[top_3_categorical_accuracy])
    model.load_weights('/home/palm/PycharmProjects/plant_d/keras/weights/try-1.h5')
    print(model.summary())

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    directory = '/home/palm/PycharmProjects/DATA/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/images/'
    out = []
    c = 0
    for imname in os.listdir(directory):
        sys.stdout.write(str(c))
        sys.stdout.flush()
        c += 1
        img = Image.open(directory+imname).resize((299, 299))
        img = np.array(img, dtype='uint8')
        img = np.reshape(img, (1, 299, 299, 3))
        img /= 127.5
        img -= 1
        prd = model.predict(img)
        out.append({'image_id': imname,
                    'disease_class': np.argmax(prd[0])})
    with open('/home/palm/PycharmProjects/plant_d/keras/prd/incp1.json', 'w') as wr:
        json.dump(out, wr)
