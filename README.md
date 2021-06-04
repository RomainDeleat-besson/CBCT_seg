# CBCT_seg

Contributors: Celia Le, Romain Deleat-Besson, Loris Bert (NYU)

Scripts for MandSeg and RootCanalSeg projects

## Prerequisites

python 3.7.9 with the librairies:

**Main librairies:**

> tensorflow==2.4.1 \
> tensorflow-gpu==2.4.0 \
> Pillow==7.2.0 \
> numpy==1.19.5 \
> itk==5.2.0 

**Detailed librairies**

> requirements.txt

## What is it?

CBCT_seg is a tool for CBCT segmentation based on a machine learning approach.

The Convolutional Neural Network (CNN) used is a 2D U-Net.

It takes several CBCT scans in input, with the extensions: .nii | nii.gz, .gipl | .gipl.gz, .nrrd

## Running the code

**Prediction**

To run the prediction algorithm, run the folowing command line:

- bash src/sh/main_prediction.sh PARAMETERS

```
the input parameters are:

--dir_src                 Folder containing the scripts.
--dir_input               Folder containing the scans to segment.
--dir_output              Folder to save the postprocessed images

the optionnal parameters are:

--width                   Width of the images
--height                  Height of the images
--tool_name               Tool name [MandSeg | RCSeg]
-h|--help                 Print this Help.
```

**Training**

To run the training algorithm, run the folowing command line (main_training_MandSeg.sh and main_training_RCSeg.sh have the same parameters but not the same values in the optionnal parameters):

- bash src/sh/main_training_MandSeg.sh PARAMETERS

```
the input parameters are:

--dir_project             Folder containing the project.
--dir_src                 Folder containing the scripts.
--dir_data                Folder containing all the data of the project.
--dir_database            Folder containing the database for the training/testing.
--dir_cv                  Folder to save the cross-validation folds.
--dir_test_preproc        Folder to save the preprocessed testing images
--dir_train_preproc       Folder to save the preprocessed training images
--dir_model               Folder to save the models
--dir_log                 Folder to save the logs of the model
--dir_test_predict        Folder to save the predicted testing images
--dir_train_predict       Folder to save the predicted training images
--dir_test_postproc       Folder to save the postprocessed testing images
--dir_train_postproc      Folder to save the postprocessed training images

the optionnal parameters are:

--cv_folds                Number of folds for the cross validation.
--testing_percentage      Percentage of images to keep for testing
--min_percentage          Min percentage to threshold images for preprocessing
--max_percentage          Max percentage to threshold images for preprocessing
--model_name              Name of the model
--epochs                  Number of epochs for training the models
--save_frequence          Frequence of saving the models
--width                   Width of the images
--height                  Height of the images
--learning_rate           Learning rate
--batch_size              Batch size
--NumberFilters           Number of filters
--dropout                 Dropout
--num_epoch               Number of the epoch of the model to select for the prediction
--tool_name               Name of the tool used
--out_metrics_val         File to save the evaluation metrics of the models on the validation set
--out_metrics_testing     File to save the evaluation metrics of the models on the testing set
-h|--help                 Print this Help.
```

**Docker**

*MandSeg:*

You can get and run the MandSeg docker image by running the folowing command line:

- docker pull dcbia/mandseg:latest

- docker run --rm -v */my/input/folder*:/app/scans mandseg:latest bash /app/src/sh/main_prediction.sh --dir_src /app/src --dir_input /app/scan --path_model /app/model/*ModelName* --min_percentage 30 --max_percentage 90 --width 256 --height 256 --tool_name MandSeg

*RCSeg:*

You can get and run the MandSeg docker image by running the folowing command line:

- docker pull dcbia/rcseg:latest

- docker run --rm -v */my/input/folder*:/app/scans rcseg:latest bash /app/src/sh/main_prediction.sh --dir_src /app/src --dir_input /app/scan --path_model /app/model/*ModelName* --min_percentage 55 --max_percentage 90 --width 512 --height 512 --tool_name RCSeg

### Creation of the folds for the cross validation

- python3 src/py/CV_folds.py 

input: folder containing all the scans and segmentations

output: folder containing the scans and segmentations divided into training/testing. The training folder is also divided into the number of folds wanted.

Takes a folder, searches into all the subfolders and seperates the scans and the segmentations. Moves thoses files into the output folder. A training folder and a testing folder are created (the training folder is devided into the specified number of folds). The percentage or number of files for testing is selected randomly inside each folder, according to the propotion of files in each of them, to prevent class imbalance.

```
usage: CV_folds.py [-h] --dir DIR --out OUT [--cv_folds CV_FOLDS]
                   [--testing_number TESTING_NUMBER | --testing_percentage TESTING_PERCENTAGE]

Creation of the cross-validation folders

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir DIR             Input directory with 3D images (default: None)

Output parameters:
  --out OUT             Output directory (default: None)
  --cv_folds CV_FOLDS   Number of folds to create (default: 10)
  --testing_number TESTING_NUMBER
                        Number of scans to keep for testing (default: 1)
  --testing_percentage TESTING_PERCENTAGE
                        Percentage of scans to keep for testing (default: 20)
```

### Pre-Processing

- python3 src/py/PreProcess.py 

input: 3D CBCT scans (.nii | nii.gz, .gipl | .gipl.gz, .nrrd)

output: 2D .png slices from the scans

Takes a single image or a directory, performs contrast adjustment and deconstructs the 3D scan into 2D slices

```
usage: PreProcess.py [-h] (--image IMAGE | --dir DIR)
                     [--desired_width DESIRED_WIDTH]
                     [--desired_height DESIRED_HEIGHT]
                     [--min_percentage MIN_PERCENTAGE]
                     [--max_percentage MAX_PERCENTAGE] [--out OUT]

Pre-processing

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --image IMAGE         Input 3D image (default: None)
  --dir DIR             Input directory with 3D images (default: None)

Resizing parameters:
  --desired_width DESIRED_WIDTH
  --desired_height DESIRED_HEIGHT

Contrast parameters:
  --min_percentage MIN_PERCENTAGE
  --max_percentage MAX_PERCENTAGE

Output parameters:
  --out OUT             Output directory (default: None)
```

- python3 src/py/labels_preprocess.py 

input: 3D CBCT labels (.nii | nii.gz, .gipl | .gipl.gz, .nrrd)

output: 2D .png slices from the labels

Takes a single label or a directory and deconstructs the 3D scan into 2D slices

```
usage: labels_preprocess.py [-h] (--image IMAGE | --dir DIR)
                            [--desired_width DESIRED_WIDTH]
                            [--desired_height DESIRED_HEIGHT] [--out OUT]

Label pre-processing

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --image IMAGE         Input 3D label (default: None)
  --dir DIR             Input directory with 3D labels (default: None)

Resizing parameters:
  --desired_width DESIRED_WIDTH
  --desired_height DESIRED_HEIGHT

Output parameters:
  --out OUT             Output directory of the label slices (default: None)
```

### Training

#### Data augmentation

The data augmentation used in the training applies random rotation, shift, shear and zoom.

- python3 src/py/heat_map.py 

input: Database that contains the labels

output: Save the the heat map

Visualisation of the data augmentation that is applied on your dataset

```
usage: heat_map.py [-h] --dir_database DIR_DATABASE [--width WIDTH]
                   [--height HEIGHT] [--out OUT]

Visualization of the data augmentation

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir_database DIR_DATABASE
                        Input dir of the labels (default: None)

label parameters:
  --width WIDTH
  --height HEIGHT

Output parameters:
  --out OUT             Output file (default: None)
```

#### Training  model

The neural network choosen is a U-Net architecture. The loss function is the *BinaryCrossentropy*. The metrics monitered during the training were the *Recall, Precision and AUC* metrics. 

- python3 src/py/training_Seg.py 

input: The scans and labels for the training

output: The model

The algorithm takes as an input the training fold. It will use one of the cross validation folder as a validation set and the rest as a training set. The *save_frequence* parameter allow you to save the model at specific epochs. 

```
usage: training_Seg.py [-h] --dir_train DIR_TRAIN --val_folds VAL_FOLDS
                       [VAL_FOLDS ...] --save_model SAVE_MODEL --log_dir
                       LOG_DIR [--model_name MODEL_NAME] [--epochs EPOCHS]
                       [--save_frequence SAVE_FREQUENCE] [--width WIDTH]
                       [--height HEIGHT] [--batch_size BATCH_SIZE]
                       [--learning_rate LEARNING_RATE]
                       [--number_filters NUMBER_FILTERS] [--dropout DROPOUT]

Training a neural network

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir_train DIR_TRAIN
                        Input training folder (default: None)
  --val_folds VAL_FOLDS [VAL_FOLDS ...]
                        Fold of the cross-validation to keep for validation
                        (default: None)
  --save_model SAVE_MODEL
                        Directory to save the model (default: None)
  --log_dir LOG_DIR     Directory for the logs of the model (default: None)

training parameters:
  --model_name MODEL_NAME
                        Name of the model (default: CBCT_seg_model)
  --epochs EPOCHS       Number of epochs (default: 20)
  --save_frequence SAVE_FREQUENCE
                        Epoch frequence to save the model (default: 5)
  --width WIDTH
  --height HEIGHT
  --batch_size BATCH_SIZE
                        Batch size value (default: 32)
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.0001)
  --number_filters NUMBER_FILTERS
                        Number of filters (default: 32)
  --dropout DROPOUT     Dropout (default: 0.1)
```

### Prediction

- python3 src/py/predict_Seg.py 

input: 2D slices form CBCT scans

output: 2D slices predicted by the model

Takes 2D slices as an input from a CBCT scan and output the label predicted for each slices.

```
usage: predict_Seg.py [-h] --dir_predict DIR_PREDICT [--width WIDTH]
                      [--height HEIGHT] --load_model LOAD_MODEL --out OUT

Prediction

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir_predict DIR_PREDICT
                        Input dir to be predicted (default: None)

Predict parameters:
  --width WIDTH
  --height HEIGHT
  --load_model LOAD_MODEL
                        Path of the trained model (default: None)

Output parameters:
  --out OUT             Output directory (default: None)
```

### Post-Processing

- python3 src/py/PostProcess.py 

input: directory with .png slices

output: directory with reconstructed 3D image

Takes an input directory containing .png slices and reconstructs the 3D image. Saves it as .nrrd.
Post-processing will be added to this function.

```
usage: PostProcess.py [-h] --dir DIR --original_dir ORIGINAL_DIR [--tool TOOL]
                      --out OUT

Post-processing

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir DIR             Input directory with 2D images (default: None)
  --original_dir ORIGINAL_DIR
                        Input directory with original 3D images (default:
                        None)

Output parameters:
  --tool TOOL           Name of the tool used (default: RCSeg)
  --out OUT             Output directory (default: None)
```

### Evaluation of the trained models

- python3 src/py/metrics.py 

input: either a 3D file and its ground truth, or a directory of 3D files and their ground truths.

output: excel file containing the mean evaluation metrics (AUC, F1 score, Accuracy, Sensitivity, Precision) of the input files.

Compares the prediction made by the algorithm to the ground truth to evaluate the performances of the trained model.
If the output excel file already exist, it adds the results into a new line.

```
usage: metrics.py [-h] (--pred_img PRED_IMG | --pred_dir PRED_DIR)
                  (--groundtruth_img GROUNDTRUTH_IMG | --groundtruth_dir GROUNDTRUTH_DIR)
                  --out OUT [--tool TOOL] [--model_name MODEL_NAME]
                  [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                  [--learning_rate LEARNING_RATE]
                  [--number_filters NUMBER_FILTERS] [--cv_fold CV_FOLD]

Evaluation metrics

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --pred_img PRED_IMG   Input predicted reconstructed 3D image (default: None)
  --pred_dir PRED_DIR   Input directory with predicted reconstructed 3D images
                        (default: None)
  --groundtruth_img GROUNDTRUTH_IMG
                        Input original 3D images (ground truth) (default:
                        None)
  --groundtruth_dir GROUNDTRUTH_DIR
                        Input directory with original 3D images (ground truth)
                        (default: None)

Output parameters:
  --out OUT             Output filename (default: None)

Training parameters:
  --tool TOOL           Name of the tool used (default: MandSeg)
  --model_name MODEL_NAME
                        name of the model (default: CBCT_seg_model)
  --epochs EPOCHS       name of the model (default: 20)
  --batch_size BATCH_SIZE
                        batch_size value (default: 16)
  --learning_rate LEARNING_RATE
  --number_filters NUMBER_FILTERS
  --cv_fold CV_FOLD     number of the cross-validation fold (default: 1)
```

