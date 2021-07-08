import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models import *
from utils import *


def main(args):
    InputDir = args.dir_database
    out = args.out
    InputdirLabel = [os.path.join(InputDir,fold,'Segs') for fold in os.listdir(InputDir) if not fold.startswith(".")]

    width = args.width
    height = args.height

    print("Loading paths...")
    # Input files and labels
    label_paths = sorted([file for file in [os.path.join(dir, fname) for dir in InputdirLabel for fname in os.listdir(dir)] if not os.path.basename(file).startswith(".")])

    print("Pre-processing...")
    # Read and process the input files
    y_train    = np.array([ProcessDataset(path, label_paths, width, height, label=True) for path in label_paths])
    y_train = np.reshape(y_train, y_train.shape+(1,))
    print("shape label:", np.shape(y_train))
    heat_map_true = np.sum(y_train, axis=0)
    
    dataset_label = tf.data.Dataset.from_tensor_slices(y_train)
    dataset_label = dataset_label.map(augment_heat_map)
    heat_map_aug = list(dataset_label.as_numpy_iterator())

    heat_map_aug = np.sum(heat_map_aug, axis=0)
    heat_map_aug = np.squeeze(heat_map_aug)
    print("shape heat_map:", np.shape(heat_map_aug))

    fig = plt.figure(1)
    fig.add_subplot(1, 2, 1)
    plt.imshow(heat_map_true, cmap='hot', interpolation='nearest')
    fig.add_subplot(1, 2, 2)
    plt.imshow(heat_map_aug, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig(out)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Visualization of the data augmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_path = parser.add_argument_group('Input files')
    input_path.add_argument('--dir_database', type=str, help='Input dir of the labels', required=True)

    param = parser.add_argument_group('label parameters')
    param.add_argument('--width', type=int, default=512, help="width of the images")
    param.add_argument('--height', type=int, default=512, help="height of the images")

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output file', required=True)
     
    args = parser.parse_args()

    main(args)













