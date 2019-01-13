import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers


if __name__ == '__main__':
    # Image directory.
    train_dir = "./data"

    batch_size = 5
    validation_split = 0.3
    img_height = 150
    img_width = 150
    epochs = 10

    # Optional additional image augmentation with ImageDataGenerator.
    input_imgen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0,
        shear_range=0.05,
        zoom_range=0,
        brightness_range=(1, 1.3),
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split)

    train_generator = input_imgen.flow_from_directory(train_dir,
                                                      target_size=(
                                                          img_height, img_width),
                                                      class_mode="categorical",
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      subset='training')
    validation_generator = input_imgen.flow_from_directory(train_dir,
                                                           target_size=(
                                                               img_height, img_width),
                                                           class_mode="categorical",
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           subset='validation')

    print('training steps: ', train_generator.samples // batch_size)
    print('validation steps: ', validation_generator.samples // batch_size)

    # Build a model for transfer learning.

    # Load a pre-trained conv base model.
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(img_height, img_width, 3))

    # Add classification layers.
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(4, activation='sigmoid'))

    # Freeze the cov base model,
    # only update classification layers weights during training.
    conv_base.trainable = False
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    model.summary()

    # Start the traning.
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs)
