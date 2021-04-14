import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models import *
from utils import *


def main(args):
    InputdirLabel = args.dir_label


    print("Loading paths...")
    # Input files and labels
    label_paths    = sorted([os.path.join(InputdirLabel, fname) for fname in os.listdir(InputdirLabel) if not fname.startswith(".")])

    print("Pre-processing...")
    # Read and process the input files
    y_train    = np.array([Array_2_5D(path, label_paths, 512, 512, 1, label=True) for path in label_paths])



    print("shape label:", np.shape(y_train))
    heat_map_true = np.sum(y_train, axis=0)

    dataAug = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='constant'),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.05, width_factor=0.05, fill_mode='constant'),
    ])

    y_train_aug = dataAug(y_train)
    heat_map_aug = np.sum(y_train_aug, axis=0)
    print("shape heat_map:", np.shape(heat_map_aug))


    fig = plt.figure(1)
    fig.add_subplot(1, 2, 1)
    plt.imshow(heat_map_true, cmap='hot', interpolation='nearest')
    fig.add_subplot(1, 2, 2)
    plt.imshow(heat_map_aug, cmap='hot', interpolation='nearest')
    plt.show()




if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Training a neural network', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    training_path = parser.add_argument_group('Paths for the training')
    training_path.add_argument('--dir_label', type=str, help='Input dir for the labeling folder', required=True)

    args = parser.parse_args()

    main(args)













