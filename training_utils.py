import pandas as pd
from IPython.core.display_functions import display

x_col_name = 'image_name'
y_col_name = 'count'


def load_generators(training_df, validation_df, dataset_path, batch_size, img_size):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    display(training_df)
    print(x_col_name)
    print(y_col_name)

    train_generator = datagen.flow_from_dataframe(
        training_df,
        dataset_path + '/frames/frames/',
        x_col=x_col_name,
        y_col=y_col_name,
        class_mode='raw',
        target_size=img_size,
        batch_size=batch_size,
    )

    validation_generator = datagen.flow_from_dataframe(
        validation_df,
        dataset_path + '/frames/frames/',
        x_col=x_col_name,
        y_col=y_col_name,
        class_mode='raw',
        target_size=img_size,
        batch_size=batch_size,
    )

    return (train_generator, validation_generator)

