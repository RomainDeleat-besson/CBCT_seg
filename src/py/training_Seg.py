import argparse
import datetime
import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

tf.config.run_functions_eagerly(True)

from models import *
from utils import *


def remove_empty_slices(img, label, ratio=0):
    """
    if ratio = 0, keep all the slices of the scan and label
    if ratio = 1, remove all the slices that don't have labels on it (of the scan and label)
    """
    L = []
    for i in range(img.shape[0]):
        if np.count_nonzero(label[i]) == 0:
            L.append(i)
            
    L = np.array(L)
    np.random.shuffle(L)
    L = L[:int(ratio*L.shape[0])].tolist()

    print("Before:", img.shape, end='   After: ')
    img = np.delete(img, L, axis=0)
    label = np.delete(label, L, axis=0) 
    print(img.shape)
    
    return img, label

def scheduler(epoch, lr):
  if epoch < 20:
    return lr
  elif epoch < 50:
    return lr * tf.math.exp(-0.1)
  else:
    return lr


def main(args):
    InputDir = args.dir_train

    if args.val_folds:
        val_folds = args.val_folds
        InputdirTrain = [os.path.join(InputDir,fold,'Scans') for fold in os.listdir(InputDir) if fold not in val_folds and not fold.startswith(".")]
        InputdirLabel = [os.path.join(InputDir,fold,'Segs') for fold in os.listdir(InputDir) if fold not in val_folds and not fold.startswith(".")]
        InputdirValTrain = [os.path.join(InputDir,fold,'Scans') for fold in val_folds]
        InputdirValLabel = [os.path.join(InputDir,fold,'Segs') for fold in val_folds]
    else:
        val_dir = args.val_dir
        InputdirTrain = [os.path.join(InputDir,'Scans')]
        InputdirLabel = [os.path.join(InputDir,'Segs')]
        InputdirValTrain = [os.path.join(val_dir,'Scans')]
        InputdirValLabel = [os.path.join(val_dir,'Segs')]

    number_epochs = args.epochs
    save_frequence = args.save_frequence
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

    print("Processing training dataset...")
    # Read and process the input files
    x_train = np.array([ProcessDataset(path, label=False) for path in input_paths])
    y_train = np.array([ProcessDataset(path, label=True) for path in label_paths])
    
    x_train, y_train = remove_empty_slices(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    
    x_train = np.reshape(x_train, x_train.shape+(1,))
    y_train = np.reshape(y_train, y_train.shape+(1,))
    
    dataset_training = create_dataset(x_train, y_train, batch_size)
    del(x_train)
    del(y_train)

    print("Processing validation dataset...")
    x_val = np.array([ProcessDataset(path,label=False) for path in ValInput_paths])
    y_val = np.array([ProcessDataset(path, label=True) for path in ValLabel_paths])

    x_val, y_val = remove_empty_slices(x_val, y_val)
    x_val, y_val = shuffle(x_val, y_val)

    x_val = np.reshape(x_val, x_val.shape+(1,))
    y_val = np.reshape(y_val, y_val.shape+(1,))
    
    dataset_validation = create_dataset(x_val, y_val, batch_size)
    del(x_val)
    del(y_val)
  
    print("Dataset info...")
    for images, labels in dataset_training.take(1):
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()

    print("=====================================================================")
    print()
    print("Inputs shape: ", np.shape(numpy_images), "min:", np.amin(numpy_images), "max:", np.amax(numpy_images), "unique:", len(np.unique(numpy_images)))
    print("Labels shape: ", np.shape(numpy_labels), "min:", np.amin(numpy_labels), "max:", np.amax(numpy_labels), "unique:", len(np.unique(numpy_labels)))
    print()
    print("=====================================================================")


    model = unet_2D(width, height, NumberFilters, dropout, lr)

    model_checkpoint = ModelCheckpoint(savedModel, monitor='loss',verbose=1, period=save_frequence)
    log_dir = os.path.join(logPath,args.model_name+"_"+datetime.datetime.now().strftime("%Y_%d_%m-%H:%M:%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
    
    if args.learning_rate_schedular:
        LR_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks_list = [model_checkpoint, tensorboard_callback, LR_callback]
    else:
        callbacks_list = [model_checkpoint, tensorboard_callback]

    model.fit(
        dataset_training,
        epochs=number_epochs,
        validation_data=dataset_validation,
        verbose=2,
        callbacks=callbacks_list,
    )



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training a neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    training_path = parser.add_argument_group('Input files')
    training_path.add_argument('--dir_train', type=str, help='Input training folder', required=True)
    training_path.add_argument('--save_model', type=str, help='Directory to save the model', required=True)
    training_path.add_argument('--log_dir', type=str, help='Directory for the logs of the model', required=True)

    validation_files = parser.add_mutually_exclusive_group(required=True)
    validation_files.add_argument('--val_folds', type=str, nargs="+", help='Fold of the cross-validation to keep for validation')
    validation_files.add_argument('--val_dir', type=str, help='')
    
    training_parameters = parser.add_argument_group('training parameters')
    training_parameters.add_argument('--model_name', type=str, help='Name of the model', default='CBCT_seg_model')
    training_parameters.add_argument('--epochs', type=int, help='Number of epochs', default=20)
    training_parameters.add_argument('--save_frequence', type=int, help='Epoch frequence to save the model', default=5)
    training_parameters.add_argument('--learning_rate_schedular', type=bool, help='Set the LRS', default=False)
    training_parameters.add_argument('--width', type=int, default=512)
    training_parameters.add_argument('--height', type=int, default=512)
    training_parameters.add_argument('--batch_size', type=int, help='Batch size value', default=32)
    training_parameters.add_argument('--learning_rate', type=float, help='Learning rate', default=0.0001)
    training_parameters.add_argument('--number_filters', type=int, help='Number of filters', default=32)
    training_parameters.add_argument('--dropout', type=float, help='Dropout', default=0.1)
       
    args = parser.parse_args()

    main(args)





