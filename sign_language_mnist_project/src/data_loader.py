from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir='data/', img_size=(64, 64), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator
