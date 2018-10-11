from keras import layers as kl, models as km, optimizers as ko, callbacks as kc
from keras import applications as ka
from keras.preprocessing.image import ImageDataGenerator, Sequence
import numpy as np

rootpath = '/root/palm/DATA/plant'
dirs = ['AgriculturalDisease_validationset', 'AgriculturalDisease_trainingset']
filrs = ['AgriculturalDisease_validation_annotations.json',
         'AgriculturalDisease_train_annotations.json']
dests = ['validate', 'train']


class Gen(Sequence):
    def __init__(self, a_gen, b_gen, batch_size):
        self.a_gen = a_gen  # plant
        self.b_gen = b_gen  # imagenet
        self.batch_size = batch_size

    def __len__(self):
        return int(min(self.a_gen.__len__(), self.b_gen.__len__()) / self.batch_size)

    def __getitem__(self, index):
        x_batch = np.zeros((self.batch_size, self.a_gen.target_size[0], self.a_gen.target_size[1], 3))
        y_batch = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            idx = index * self.batch_size + i
            if idx >= min(self.a_gen.__len__(), self.b_gen.__len__()):
                break
            if np.random.rand() > 0.5:
                x_batch[i], _ = self.a_gen[idx]
                y_batch[i] = 0
            else:
                x_batch[i], _ = self.b_gen[idx]
                y_batch[i] = 1
        return x_batch, y_batch


def getmodel():
    m = ka.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

    x = m.get_layer('global_average_pooling2d_1').output
    x = kl.Dense(1024, activation='relu')(x)
    x = kl.Dense(1, activation='sigmoid')(x)
    m = km.Model(m.input, x)
    return m


if __name__ == '__main__':
    no = 'xcp1'
    model = getmodel()
    model.compile(optimizer=ko.sgd(lr=1e-2, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['acc'],
                  )
    checkpoint = kc.ModelCheckpoint('weights/discrmn-{}.h5'.format(no),
                                    monitor='val_acc',
                                    mode='max',
                                    save_best_only=1,
                                    save_weights_only=1)
    train_generator = ImageDataGenerator(rotation_range=180,
                                         channel_shift_range=0.2,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         preprocessing_function=ka.inception_resnet_v2.preprocess_input)
    test_generator = ImageDataGenerator(preprocessing_function=ka.inception_resnet_v2.preprocess_input)
    a_train_datagen = train_generator.flow_from_directory(f'{rootpath}/{dests[1]}',
                                                          target_size=(299, 299),
                                                          batch_size=1,
                                                          )
    a_test_datagen = test_generator.flow_from_directory(f'{rootpath}/{dests[0]}',
                                                        target_size=(299, 299),
                                                        batch_size=1,
                                                        )
    b_train_datagen = train_generator.flow_from_directory('/root/datasets/imagenet2016/ILSVRC/Data/CLS-LOC/train/',
                                                          target_size=(299, 299),
                                                          batch_size=1,
                                                          )
    b_test_datagen = test_generator.flow_from_directory('/root/palm/PycharmProjects/DATA/',
                                                        target_size=(299, 299),
                                                        batch_size=1
                                                        )
    train_datagen = Gen(a_train_datagen, b_train_datagen, 32)
    test_datagen = Gen(a_test_datagen, b_test_datagen, 32)
    f = model.fit_generator(train_datagen,
                            epochs=5,
                            validation_data=test_datagen,
                            callbacks=[checkpoint]
                            )
