import pandas as pd
from IPython.core.display_functions import display
import numpy as np
import matplotlib.pyplot as plt


x_col_name = 'image_name'
y_col_name = 'count'


def load_generators(training_df, validation_df, dataset_path, batch_size, img_size, img_path='/frames/frames/'):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    display(training_df)
    print(x_col_name)
    print(y_col_name)

    train_generator = datagen.flow_from_dataframe(
        training_df,
        dataset_path + img_path,
        x_col=x_col_name,
        y_col=y_col_name,
        class_mode='raw',
        target_size=img_size,
        batch_size=batch_size,
    )

    validation_generator = datagen.flow_from_dataframe(
        validation_df,
        dataset_path + img_path,
        x_col=x_col_name,
        y_col=y_col_name,
        class_mode='raw',
        target_size=img_size,
        batch_size=batch_size,
    )

    return train_generator, validation_generator


def load_single_generator(df, dataset_path, batch_size, img_size, img_path='/frames/frames/', debug=False):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1. / 255,
    )
    if debug:
        display(df)
        print(x_col_name)
        print(y_col_name)

    print(dataset_path + img_path)
    generator = datagen.flow_from_dataframe(
        df,
        dataset_path + img_path,
        x_col=x_col_name,
        y_col=y_col_name,
        class_mode='raw',
        target_size=img_size,
        batch_size=batch_size,
        )

    return generator

def history_metric_evaluation(history_single, display_epochs = (0, 100)):
    print(f'Min mae: {np.min(history_single.history["mae"])}')
    print(f'Min val_mae: {np.min(history_single.history["val_mae"])}')
    print(f'Min loss: {np.min(history_single.history["loss"])}')
    print(f'Min val_loss: {np.min(history_single.history["val_loss"])}')

    plt.plot(history_single.history['mae'][display_epochs[0]:display_epochs[1]])
    plt.plot(history_single.history['val_mae'][display_epochs[0]:display_epochs[1]])
    plt.title('model mean squared')
    plt.ylabel('mean squared')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(history_single.history['loss'][display_epochs[0]:display_epochs[1]])
    plt.plot(history_single.history['val_loss'][display_epochs[0]:display_epochs[1]])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def k_fold_metric_evaluation(history):
    print(history)
    mea_results = [np.min(x.history['mae']) for x in history]
    val_mea_results = [np.min(x.history['val_mae']) for x in history]
    loss_results = [np.min(x.history['loss']) for x in history]
    val_loss_results = [np.min(x.history['val_loss']) for x in history]
    print(mea_results)
    print(f'Mean mae: {np.mean(mea_results)}')
    print(f'Mean val_mae: {np.mean(val_mea_results)}')
    print(f'Mean loss: {np.mean(loss_results)}')
    print(f'Mean val_loss: {np.mean(val_loss_results)}')

