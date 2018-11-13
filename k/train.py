from keras import applications as ka
from keras import layers as kl, models as km, optimizers as ko, callbacks as kc
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import json
import keras
import numpy as np
import os
from PIL import Image as pil_image
import tensorflow as tf

rootpath = '/root/palm/PycharmProjects/plant_d'
dests = ['validate', 'train']
_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
}
# These methods were only introduced in version 3.4.0 (2016).
if hasattr(pil_image, 'HAMMING'):
    _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
if hasattr(pil_image, 'BOX'):
    _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
# This method is new in version 1.1.3 (2013).
if hasattr(pil_image, 'LANCZOS'):
    _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def lr_reduce(epoch):
    if epoch < 20:
        return min(0.01 + 0.004 * epoch, 0.1)
    elif epoch < 45:
        return 1e-2
    else:
        return 1e-3


def getmodel():
    m = ka.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', pooling='avg')

    x = m.get_layer('avg_pool').output
    x = kl.Dense(61, activation='softmax')(x)
    m = km.Model(m.input, x)
    return m


class DotDict(dict):
    def __getattr__(self, name):
        return self[name]


if __name__ == '__main__':
    no = 'incp_resnet_1'
    args = DotDict({
        'batch_size': 6,
        'batch_mul': 4,
        'val_batch_size': 10,
        'cuda': True,
        'logdir': os.path.join(rootpath, 'k/log', no),
        'model': '',
        'train_plot': False,
        'epochs': [180],
        'try_no': '3_pnasnet',
        'imsize': [331],
        'imsize_l': [350],
        'traindir': '/root/palm/DATA/plant/train',
        'valdir': '/root/palm/DATA/plant/validate',
        'workers': 16,
        'resume': False,
    })

    model = getmodel()

    model.compile(optimizer=ko.sgd(lr=0.1, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    train_generator = ImageDataGenerator(rotation_range=0.12,
                                         width_shift_range=0.12,
                                         height_shift_range=0.12,
                                         shear_range=0.1,
                                         zoom_range=0.1,
                                         horizontal_flip=True,
                                         preprocessing_function=ka.inception_resnet_v2.preprocess_input)
    test_generator = ImageDataGenerator(preprocessing_function=ka.inception_resnet_v2.preprocess_input)
    train_datagen = train_generator.flow_from_directory(args.traindir,
                                                        target_size=(299, 299), )
    test_datagen = test_generator.flow_from_directory(args.valdir,
                                                      target_size=(299, 299),
                                                      )

    reduce_lr = kc.LearningRateScheduler(lr_reduce, verbose=1)
    checkpoint = kc.ModelCheckpoint(f'{rootpath}/k/weights/try-{no}.h5',
                                    monitor='val_acc',
                                    mode='max',
                                    save_best_only=1,
                                    save_weights_only=1)
    tensorboard = kc.TensorBoard(log_dir=args.logdir,
                              histogram_freq=1,
                              write_graph=True,
                              write_images=False)

    f = model.fit_generator(train_datagen,
                            epochs=60,
                            validation_data=test_datagen,
                            callbacks=[reduce_lr, checkpoint, tensorboard]
                            )
    # with open(f'{rootpath}/log/k_{no}.json', 'w') as wr:
    #     json.dump(f.history, wr)
