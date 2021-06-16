import argparse
import datetime
import os

import numpy as np
import tensorflow as tf

from utils import *


def main(args):
    Inputdir = args.dir_predict
    load_model = args.load_model
    out = args.out
    
    width = args.width
    height = args.height

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

    if not os.path.exists(out):
        os.makedirs(out)

    print("Loading data...")
    input_paths = sorted([os.path.join(Inputdir, fname) for fname in os.listdir(Inputdir) if not fname.startswith(".")])
    images = np.array([ProcessDataset(path, label=False) for path in input_paths])

    model = tf.keras.models.load_model(load_model)

    print("Prediction & Saving...")
    for i in range(np.shape(images)[0]):
        image = np.reshape(images[i], (1,)+images[i].shape)
        prediction = model.predict(image)
        
        if np.amax(prediction)>0:
            prediction = (prediction-np.amin(prediction))/(np.amax(prediction)-np.amin(prediction))
            prediction *= 255

        outputFilename = os.path.join(out, os.path.basename(input_paths[i]))
        prediction = np.reshape(prediction, (width, height, 1))
        Save_png(outputFilename, prediction)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Prediction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prediction_path = parser.add_argument_group('Input files')
    prediction_path.add_argument('--dir_predict', type=str, help='Input dir to be predicted', required=True)
    
    predict_parameters = parser.add_argument_group('Predict parameters')
    predict_parameters.add_argument('--width', type=int, default=512)
    predict_parameters.add_argument('--height', type=int, default=512)
    predict_parameters.add_argument('--load_model', type=str, help='Path of the trained model', required=True)  

    ouput = parser.add_argument_group('Output parameters')
    ouput.add_argument('--out', type=str, help='Output directory', required=True) 

    args = parser.parse_args()

    main(args)








