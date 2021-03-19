import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from models import *
from utils import *



def main(args):

    InputdirTrain = args.dir_train
    InputdirLabel = args.dir_label
    InputdirValTrain = args.dir_valTrain
    InputdirValLabel = args.dir_valLabel

    neighborhood = args.neighborhood
    width = args.width
    height = args.height
    NumberFilters = args.number_filters
    dropout = args.dropout

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
    input_paths    = sorted([os.path.join(InputdirTrain, fname) for fname in os.listdir(InputdirTrain) if not fname.startswith(".")])
    label_paths    = sorted([os.path.join(InputdirLabel, fname) for fname in os.listdir(InputdirLabel) if not fname.startswith(".")])

    # Folder with the validations scans and labels
    ValInput_paths = sorted([os.path.join(InputdirValTrain, fname) for fname in os.listdir(InputdirValTrain) if not fname.startswith(".")])
    ValLabel_paths = sorted([os.path.join(InputdirValLabel, fname) for fname in os.listdir(InputdirValLabel) if not fname.startswith(".")])


    print("Pre-processing...")
    # Read and process the input files
    inputs    = np.array([Array_2_5D(path, input_paths, width, height, neighborhood,label=False) for path in input_paths])
    labels    = np.array([Array_2_5D(path, label_paths, width, height, neighborhood,label=True) for path in label_paths])
    ValInputs = np.array([Array_2_5D(path, ValInput_paths, width, height, neighborhood,label=False) for path in ValInput_paths])
    ValLabels = np.array([Array_2_5D(path, ValLabel_paths, width, height, neighborhood,label=True) for path in ValLabel_paths])


    if args.display:
        print("TODO")
        # image = inputs[0]
        # Plot_slices(1, 3, width, height, image[:, :, :])


    print("Create dataset...")
    x_train = inputs
    y_train = labels
    x_val = ValInputs
    y_val = ValLabels

    batch_size = args.batch_size
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Define data loaders.
    train_loader      = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_dataset = (
        train_loader.shuffle(len(x_train))
        .batch(batch_size)
        .prefetch(AUTOTUNE))

    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .batch(batch_size)
        .prefetch(AUTOTUNE))


    print("Training...")
    print("=====================================================================")
    print()
    print("Inputs shape:     ", np.shape(x_train))
    print("Labels shape:     ", np.shape(y_train))
    print("Val inputs shape: ", np.shape(x_val))
    print("Val labels shape: ", np.shape(y_val))
    print()
    print("=====================================================================")


    model = unet_2D(width, height, neighborhood, NumberFilters, dropout)
    # model = unet_2D_deeper(width, height, neighborhood, 32, dropout)
    # model = unet_2D_larger(width, height, neighborhood, 64, dropout)

    # model_checkpoint = ModelCheckpoint(savedModel, monitor='loss',verbose=1, period=5)
    # log_dir = logPath+datetime.datetime.now().strftime("%Y_%d_%m-%H:%M:%S")
    # tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
    # callbacks_list = [model_checkpoint, tensorboard_callback]

    # epochs = 20
    # model.fit(
    #     train_dataset,
    #     validation_data=validation_dataset,
    #     epochs=epochs,
    #     shuffle=True,
    #     verbose=2,
    #     callbacks=callbacks_list,
    # )



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training a neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    training_path = parser.add_argument_group('Paths for the training')
    training_path.add_argument('--dir_train', type=str, help='Input dir for the training folder', required=True)
    training_path.add_argument('--dir_label', type=str, help='Input dir for the labeling folder', required=True)
    training_path.add_argument('--dir_valTrain', type=str, help='Input dir for the validation folder', required=True)
    training_path.add_argument('--dir_valLabel', type=str, help='Input dir for the validation folder', required=True)
    training_path.add_argument('--save_model', type=str, help='Directory to save the model', required=True)
    training_path.add_argument('--log_dir', type=str, help='Directory for the logs of the model', required=True)
    
    training_parameters = parser.add_argument_group('Universal ID parameters')
    training_parameters.add_argument('--model_name', type=str, help='name of the model', default='RootSlices_model')
    training_parameters.add_argument('--width', type=int, help='batch_size value', default=512)
    training_parameters.add_argument('--height', type=int, help='batch_size value', default=512)
    training_parameters.add_argument('--batch_size', type=int, help='batch_size value', default=32)
    training_parameters.add_argument('--number_filters', type=int, help='batch_size value', default=64)
    training_parameters.add_argument('--dropout', type=float, help='batch_size value', default=0.1)
    training_parameters.add_argument('--neighborhood', type=int, choices=[3,5,7,9], help='neighborhood slices (3|5|7)', default=3)
 
    display = parser.add_argument_group('Universal ID parameters')
    display.add_argument('--display', type=int, help='', default=0)
       

    args = parser.parse_args()

    main(args)





