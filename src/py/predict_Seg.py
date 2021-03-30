import argparse
import datetime
import shutil
import glob
import itk
import os

import numpy as np
import tensorflow as tf

from utils import *



def main(args):

    Inputdir = args.dir_predict
    load_model = args.load_model
    out = args.out_dir
    
    neighborhood = args.neighborhood
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
    # else:
    #     shutil.rmtree(out)
    #     os.makedirs(out)


    print("Loading data...")
    input_paths = sorted([os.path.join(Inputdir, fname) for fname in os.listdir(Inputdir) if not fname.startswith(".")])
    images = np.array([Array_2_5D(path, input_paths, width, height, neighborhood, label=False) for path in input_paths])

    model = tf.keras.models.load_model(load_model)

    print("Prediction & Saving...")
    for i in range(np.shape(images)[0]):
        image = np.reshape(images[i], (1,)+images[i].shape)
        prediction = model.predict(image)
        prediction[prediction<=0.5]=0
        prediction[prediction>0.5]=1
        outputFilename = os.path.join(out, os.path.basename(input_paths[i]))
        prediction = np.reshape(prediction, [s for s in prediction.shape if s != 1])
        prediction = np.reshape(prediction, prediction.shape+(1,))
        Save_png(outputFilename, prediction)





if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prediction_path = parser.add_argument_group('Paths for the prediction')
    prediction_path.add_argument('--dir_predict', type=str, help='Input dir for the training folder', required=True)
    
    predict_parameters = parser.add_argument_group('')
    predict_parameters.add_argument('--width', type=int, help='', default=512)
    predict_parameters.add_argument('--height', type=int, help='', default=512)
    predict_parameters.add_argument('--neighborhood', type=int, choices=[1,3,5,7,9], help='neighborhood slices (3|5|7)', default=3)
 
    model = parser.add_argument_group('')
    model.add_argument('--load_model', type=str, help='', required=True)  

    ouput = parser.add_argument_group('')
    ouput.add_argument('--out_dir', type=str, help='', required=True) 


    args = parser.parse_args()

    main(args)








