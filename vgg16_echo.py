import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint)

# parameters:

SZ = 224

BZ = 500

LN = 152

LR = 0.0001

DO = 0.5

RG = 50

# create the base pre-trained model
base_model = VGG16(weights= "imagenet", input_shape = (224,224,3), include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# dropout
x = Dropout(DO)(x)
# let's add a fully-connected layer
x = Dense(4096, activation='relu')(x)
x = Dropout(DO)(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(3, activation='sigmoid')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# set model compile:
model.compile(optimizer=SGD(lr=LR, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])


# sample = np.load("../raw/sample.npy")


# Checkpoint
checkpointer = ModelCheckpoint(
    filepath="model_checkpoint_{}_{}.h5".format("first", "title"),
    verbose=1,
    save_best_only=True)

# csvlogger
csv_logger = CSVLogger(
    'csv_logger_{}_{}.csv'.format("first", "title"))
# EarlyStopping
# early_stopper = EarlyStopping(monitor='val_loss',
#                               min_delta=0.001,
#                              patience=20)

# Reduce lr on plateau
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10,
                               min_lr=0.5e-5)

# image data generator:
train_datagen = ImageDataGenerator(
    rotation_range=15.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.,
    zoom_range=0.1,
    channel_shift_range=0.,
    fill_mode='nearest',
    #fill_mode = "constant",
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    preprocessing_function=None)

# train_datagen.fit(sample,augment=True)

train_generator = train_datagen.flow_from_directory(
        '/ysm-gpfs/home/hc487/project/ASDrgb3/train',
        target_size=(SZ, SZ),
        batch_size = BZ,
        shuffle = True,
        class_mode='categorical')

# save_to_dir='../raw/preview', save_prefix='cat', save_format='jpg'

test_datagen = ImageDataGenerator(
    rescale=1./255)


# test_datagen.fit(sample,augment=True)

test_generator = test_datagen.flow_from_directory(
        '/ysm-gpfs/home/hc487/project/ASDrgb3/test',
        target_size=(SZ, SZ),
        batch_size = BZ,
        shuffle = True,
        class_mode='categorical')

#
# from keras.models import load_model
# model = load_model("./model1/model_checkpoint_first_title.h5")



# save
model.fit_generator(train_generator, steps_per_epoch= 40000. / BZ, epochs= 500, validation_data = test_generator, validation_steps = 1000. / BZ, callbacks=[csv_logger, checkpointer])

#model.save("model_final.h5")
# model.fit(train, y_train, epochs=500, batch_size=20, shuffle = True,validation_data=(test, y_test),callbacks=[csv_logger, checkpointer])

