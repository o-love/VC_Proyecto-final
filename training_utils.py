import pandas as pd
from IPython.core.display_functions import display


def load_tri_image_generators(labels_df, dataset_path, batch_size, img_size, dataset_sizes):
    display(labels_df)

    (train_size, validation_size, test_size) = dataset_sizes

    if (train_size+validation_size+test_size) != len(labels_df):
        print('Dataset size is different from specified class sizes')
        exit(1)

    from tensorflow.keras.preprocessing.image import ImageDataGenerator


    training_df = labels_df[:train_size]
    validation_df = labels_df[train_size:train_size+validation_size].reset_index(drop=True)
    test_df = labels_df[train_size+validation_size:].reset_index(drop=True)


    datagen = ImageDataGenerator(
        rescale=1./255,
    )

    x_col_name = 'image_name'
    y_col_name = 'count'

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

    test_generator = datagen.flow_from_dataframe(
        test_df,
        dataset_path + '/frames/frames/',
        x_col=x_col_name,
        y_col=y_col_name,
        class_mode='raw',
        target_size=img_size,
        batch_size=batch_size,
    )