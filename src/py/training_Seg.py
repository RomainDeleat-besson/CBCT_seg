import argparse
import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from models import *
from utils import *

def remove_empty_slices(img, label):

    L = []
    for i in range(img.shape[0]):
        if np.count_nonzero(label[i]) == 0:
            L.append(i)
            
    L = np.array(L)
    np.random.shuffle(L)
    L = L[:int(0.66*L.shape[0])].tolist()

    print("Before:", img.shape, end='   After: ')
    img = np.delete(img, L, axis=0)
    label = np.delete(label, L, axis=0) 
    print(img.shape)
    
    return img, label

def map_decorator(func):
    def wrapper(*args):
        return tf.py_function(
            func=func,
            inp=[*args],
            Tout=[a.dtype for a in args])
    return wrapper

def aug_layers(x, seed):
    np.random.seed(seed)
    
    x = tf.image.rot90(x, seed%4) # Rotate 0, 90, 180, 270 degrees
    x = tf.keras.preprocessing.image.random_shift(x, 0.05, 0.05, fill_mode='constant')
    x = tf.keras.preprocessing.image.random_rotation(x, 25, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
    # x = tf.keras.preprocessing.image.random_shear(x, 10, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant')
    # x = tf.keras.preprocessing.image.random_zoom(x, (0.95, 0.95), fill_mode='constant')
    return x

@map_decorator
def augment(x, y):
    seed = random.randint(0,999999)
    # print(seed)
    
    x = aug_layers(x, seed)
    y = aug_layers(y, seed)

    return x, y



def main(args):

    InputDir = args.dir_train
    val_folds = args.val_folds

    InputdirTrain = [os.path.join(InputDir,fold,'Scans') for fold in os.listdir(InputDir) if fold not in val_folds and not fold.startswith(".")]
    InputdirLabel = [os.path.join(InputDir,fold,'Segs') for fold in os.listdir(InputDir) if fold not in val_folds and not fold.startswith(".")]
    InputdirValTrain = [os.path.join(InputDir,fold,'Scans') for fold in val_folds]
    InputdirValLabel = [os.path.join(InputDir,fold,'Segs') for fold in val_folds]

    number_epochs = args.epochs
    save_frequence = args.save_frequence
    neighborhood = args.neighborhood
    width = args.width
    height = args.height
    batch_size = args.batch_size
    NumberFilters = args.number_filters
    dropout = args.dropout
    lr = args.learning_rate

    savedModel = os.path.join(args.save_model, args.model_name+"_{epoch}.hdf5")
    logPath = args.log_dir


    # GPUs Initialization
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    print("Loading paths...")
    # Input files and labels
    input_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirTrain for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])
    label_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirLabel for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])

    # Folder with the validations scans and labels
    ValInput_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirValTrain for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])
    ValLabel_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirValLabel for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])

    print("Pre-processing...")
    # Read and process the input files
    x_train    = np.array([Array_2_5D(path, input_paths, width, height, neighborhood,label=False) for path in input_paths])
    y_train    = np.array([Array_2_5D(path, label_paths, width, height, neighborhood,label=True) for path in label_paths])
    x_val      = np.array([Array_2_5D(path, ValInput_paths, width, height, neighborhood,label=False) for path in ValInput_paths])
    y_val      = np.array([Array_2_5D(path, ValLabel_paths, width, height, neighborhood,label=True) for path in ValLabel_paths])

    x_train, y_train = remove_empty_slices(x_train, y_train)

    x_train = np.reshape(x_train, (width, height, 1))
    y_train = np.reshape(y_train, (width, height, 1))
    x_val = np.reshape(x_val, (width, height, 1))
    y_val = np.reshape(y_val, (width, height, 1))


    print("Training...")
    print("=====================================================================")
    print()
    print("Inputs shape:     ", np.shape(x_train), "min:", np.amin(x_train), "max:", np.amax(x_train), "unique:", len(np.unique(x_train)))
    print("Labels shape:     ", np.shape(y_train), "min:", np.amin(y_train), "max:", np.amax(y_train), "unique:", len(np.unique(y_train)))
    print("Val inputs shape: ", np.shape(x_val), "min:", np.amin(x_val), "max:", np.amax(x_val), "unique:", len(np.unique(x_val)))
    print("Val labels shape: ", np.shape(y_val), "min:", np.amin(y_val), "max:", np.amax(y_val), "unique:", len(np.unique(y_val)))
    print()
    print("=====================================================================")


    BATCH_SIZE = batch_size
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    dataset_training = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_training = dataset_training.map(augment, num_parallel_calls=AUTOTUNE)
    dataset_training = dataset_training.shuffle(8*BATCH_SIZE)
    dataset_training = dataset_training.batch(BATCH_SIZE)
    dataset_training = dataset_training.prefetch(AUTOTUNE)
    
    dataset_validation = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    dataset_validation = dataset_validation.map(augment, num_parallel_calls=AUTOTUNE)
    dataset_validation = dataset_validation.batch(BATCH_SIZE)
    dataset_validation = dataset_validation.prefetch(AUTOTUNE)


    model = unet_2D(width, height, neighborhood, NumberFilters, dropout, lr)
    # model = unet_2D_deeper(width, height, neighborhood, 32, dropout, lr)
    # model = unet_2D_larger(width, height, neighborhood, 64, dropout, lr)

    model_checkpoint = ModelCheckpoint(savedModel, monitor='loss',verbose=1, period=save_frequence)
    log_dir = logPath+datetime.datetime.now().strftime("%Y_%d_%m-%H:%M:%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
    callbacks_list = [model_checkpoint, tensorboard_callback]

    model.fit(
        dataset_training,
        epochs=number_epochs,
        batch_size=None,
        validation_data=dataset_validation,
        verbose=2,
        callbacks=callbacks_list,
        # x_train,
        # y_train,
        # batch_size=batch_size,
        # validation_data=(x_val, y_val),
        # epochs=number_epochs,
        # shuffle=True,
        # verbose=2,
        # callbacks=callbacks_list,
    )



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training a neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    training_path = parser.add_argument_group('Paths for the training')
    training_path.add_argument('--dir_train', type=str, help='Input training folder', required=True)
    training_path.add_argument('--val_folds', type=str, nargs="+", help='Fold of the cross-validation to keep for validation', required=True)
    training_path.add_argument('--save_model', type=str, help='Directory to save the model', required=True)
    training_path.add_argument('--log_dir', type=str, help='Directory for the logs of the model', required=True)
    
    training_parameters = parser.add_argument_group('Universal ID parameters')
    training_parameters.add_argument('--model_name', type=str, help='name of the model', default='CBCT_seg_model')
    training_parameters.add_argument('--epochs', type=int, help='name of the model', default=20)
    training_parameters.add_argument('--save_frequence', type=int, help='name of the model', default=5)
    training_parameters.add_argument('--width', type=int, help='', default=512)
    training_parameters.add_argument('--height', type=int, help='', default=512)
    training_parameters.add_argument('--batch_size', type=int, help='batch_size value', default=32)
    training_parameters.add_argument('--learning_rate', type=float, help='', default=0.0001)
    training_parameters.add_argument('--number_filters', type=int, help='', default=64)
    training_parameters.add_argument('--dropout', type=float, help='', default=0.1)
    training_parameters.add_argument('--neighborhood', type=int, choices=[1,3,5,7,9], help='neighborhood slices (3|5|7)', default=3)
 
    display = parser.add_argument_group('Universal ID parameters')
    display.add_argument('--display', type=int, help='', default=0)
       

    args = parser.parse_args()

    main(args)





