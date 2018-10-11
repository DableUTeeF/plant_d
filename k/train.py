from keras import applications as ka
from keras import layers as kl, models as km, optimizers as ko, callbacks as kc
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import json
import keras
import numpy as np
import os
from PIL import Image as pil_image
import tensorflow as tf
rootpath = '/content/drive/My Drive/plant_d'
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
    if epoch < 25:
        return 2e-2
    elif 25 <= epoch < 45:
        return 1e-3
    else:
        return 1e-1


def getmodel():
    m = ka.nasnet.NASNetMobile(include_top=True, weights='imagenet', pooling='avg')

    x = m.get_layer('global_average_pooling2d_1').output
    x = kl.Dense(1024, activation='relu')(x)
    x = kl.Dense(61, activation='softmax')(x)
    m = km.Model(m.input, x)
    return m


def pad(img):
    width, height = img.size
    if width > height:
        img = img.resize((299, int(height / width * 299)))
    else:
        img = img.resize((int(width / height * 299), 299))
    outim = pil_image.new('RGB',
                          (299, 299),
                          (0, 0, 0),
                          )
    outim.paste(img)
    if width > height:
        outim.paste(img, (0, int(height / width * 299)))
    else:
        outim.paste(img, (int(width / height * 299)+5, 0))
    return outim


def load_img(path, grayscale=False, target_size=None,
             interpolation='nearest'):
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    img = pad(img)

    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


class DIterator(DirectoryIterator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=keras.backend.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = keras.preprocessing.image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(keras.backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=keras.backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class Datagen(ImageDataGenerator):
    def __init__(self, rotation_range=45, channel_shift_range=0.0, horizontal_flip=True, vertical_flip=True,
                 padding='replica', preprocessing_function=None):
        self.padding = padding
        super().__init__(rotation_range=rotation_range,
                         channel_shift_range=channel_shift_range,
                         horizontal_flip=horizontal_flip,
                         vertical_flip=vertical_flip,
                         preprocessing_function=preprocessing_function)

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            interpolation='nearest'):
        return DIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            interpolation=interpolation)


if __name__ == '__main__':
    no = 'nasM1'
    model = getmodel()
    resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu="grpc://"+os.environ["COLAB_TPU_ADDR"])
    strategy = tf.contrib.tpu.TPUDistributionStrategy(resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    session_master = resolver.master()

    model.compile(optimizer=ko.sgd(lr=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    train_generator = ImageDataGenerator(preprocessing_function=ka.inception_resnet_v2.preprocess_input)
    test_generator = ImageDataGenerator(preprocessing_function=ka.inception_resnet_v2.preprocess_input)
    train_datagen = train_generator.flow_from_directory(f'{rootpath}/{dests[1]}',
                                                        target_size=(224, 224), )
    test_datagen = test_generator.flow_from_directory(f'{rootpath}/{dests[0]}',
                                                      target_size=(224, 224),
                                                      )

    reduce_lr = kc.LearningRateScheduler(lr_reduce, verbose=1)
    checkpoint = kc.ModelCheckpoint(f'{rootpath}/weights/try-{no}.h5',
                                    monitor='val_acc',
                                    mode='max',
                                    save_best_only=1,
                                    save_weights_only=1)

    f = model.fit_generator(train_datagen,
                            epochs=50,
                            validation_data=test_datagen,
                            callbacks=[reduce_lr, checkpoint]
                            )
    with open(f'{rootpath}/log/k_{no}.json', 'w') as wr:
        json.dump(f.history, wr)
