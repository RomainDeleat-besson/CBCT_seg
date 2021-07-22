# CBCT_seg

Authors: Deleat-Besson Romain (UoM), Le Celia (UoM)

Contributor: Bert Loris (NYU)

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

It takes several CBCT scans as inputs, with the extensions: .nii | nii.gz, .gipl | .gipl.gz, .nrrd

## Running the code

MandSeg and RCSeg algorithms have different parameters therefore 4 bash scripts were created to run the training and prediction algorithms.

**Prediction**

To run the prediction algorithm, run the folowing command line:

```
bash src/sh/main_prediction_MandSeg.sh --help
```
```
bash src/sh/main_prediction_RCSeg.sh --help
```

```
the input parameters are:

--dir_src                 Path to the Folder that contains the source code
--file_input              Scan to segment
--dir_preproc             Folder to save the preprocessed images
--dir_predicted           Folder to save the predicted images
--dir_output              Folder to save the postprocessed images

The optionnal parameters are:

--width                   Width of the images
--height                  Height of the images
--tool_name               Tool name [MandSeg | RCSeg]
--threshold               Threshold to use to binarize scans in postprocess. (-1 for otsu | [0;255] for a specific value)
-h|--help                 Print this Help
```

**Training**

To run the training algorithm, run the folowing command line:

```
bash src/sh/main_training_MandSeg.sh --help
```
```
bash src/sh/main_training_RCSeg.sh --help
```

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
--ratio                   Ration of slices outside of the region of interest to remove (value between [0;1])
--num_epoch               Number of the epoch of the model to select for the prediction
--tool_name               Name of the tool used
--threshold               Threshold to use to binarize scans in postprocess. (-1 for otsu | [0;255] for a specific value)
--out_metrics_val         File to save the evaluation metrics of the models on the validation set
--out_metrics_testing     File to save the evaluation metrics of the models on the testing set
-h|--help                 Print this Help.
```

### Docker

You can get the CBCT_seg docker image by running the folowing command line:

```
docker pull dcbia/cbct-seg:latest
```

*MandSeg:*

To run MandSeg inside the docker container, run the following command line:

```
docker run --rm -v */my/input/file*:/app/scans/$(basename */my/input/file*) -v */my/output/folder*:/app/out cbct-seg:latest main_prediction_MandSeg.sh --dir_src /app/src --file_input /app/scans/$(basename */my/input/file*) --dir_output /app/out --path_model /app/model/MandSeg_Final_35.hdf5
```

*RCSeg:*

To run RCSeg inside the docker container, run the following command line:

```
docker run --rm -v */my/input/file*:/app/scans/$(basename */my/input/file*) -v */my/output/folder*:/app/out cbct-seg:latest main_prediction_RCSeg.sh --dir_src /app/src --file_input /app/scans/$(basename */my/input/file*) --dir_output /app/out --path_model /app/model/RCSeg_Final_50.hdf5
```

### Creation of the workspace

```
python3 src/py/generate_workspace.py --help
```

input: folder containing all the scans and segmentations

output: folder containing the scans and segmentations divided into training/testing(/validation)

Takes a folder, searches into all the subfolders and seperates the scans and the segmentations. Moves thoses files into the output folder. A training folder and a testing folder are created. The training folder is divided into the specified number of folds, or, if the number of folds is 0, a validation folder can be created. The percentage or number of files for testing and validation is selected randomly inside each folder, according to the propotion of files in each of them, to prevent class imbalance.

```
usage: generate_workspace.py [-h] --dir DIR --out OUT [--cv_folds CV_FOLDS]
                             [--testing_number TESTING_NUMBER | --testing_percentage TESTING_PERCENTAGE]
                             [--validation_number VALIDATION_NUMBER | --validation_percentage VALIDATION_PERCENTAGE]

Creation of the workspace (training, validation, testing | cross validation)

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir DIR             Input directory with 3D images (default: None)

Output parameters:
  --out OUT             Output directory (default: None)
  --cv_folds CV_FOLDS   Number of folds to create (default: 8)
  --testing_number TESTING_NUMBER
                        Number of scans to keep for testing (default: None)
  --testing_percentage TESTING_PERCENTAGE
                        Percentage of scans to keep for testing (default: 20)
  --validation_number VALIDATION_NUMBER
                        Number of scans to keep for validation (default: None)
  --validation_percentage VALIDATION_PERCENTAGE
                        Percentage of scans to keep for validation (default:
                        10)
```

### Pre-Processing

```
python3 src/py/preprocess.py --help
```

input: 3D CBCT scans (.nii | nii.gz, .gipl | .gipl.gz, .nrrd)

output: 2D .png slices from the scans

Takes a single image or a directory, performs contrast adjustment and deconstructs the 3D scan into 2D slices of the specified size.

```
usage: preprocess.py [-h] (--image IMAGE | --dir DIR)
                     [--desired_width DESIRED_WIDTH]
                     [--desired_height DESIRED_HEIGHT]
                     [--min_percentage MIN_PERCENTAGE]
                     [--max_percentage MAX_PERCENTAGE] --out OUT

Pre-processing

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --image IMAGE         Input 3D image (default: None)
  --dir DIR             Input directory with 3D images (default: None)

Resizing parameters:
  --desired_width DESIRED_WIDTH
                        desired width of the images (default: 512)
  --desired_height DESIRED_HEIGHT
                        desired width of the images (default: 512)

Contrast parameters:
  --min_percentage MIN_PERCENTAGE
                        min percentage to adjust contrast of the images
                        (default: 45)
  --max_percentage MAX_PERCENTAGE
                        max percentage to adjust contrast of the images
                        (default: 90)

Output parameters:
  --out OUT             Output directory (default: None)
```

```
python3 src/py/labels_preprocess.py --help
```

input: 3D CBCT labels (.nii | nii.gz, .gipl | .gipl.gz, .nrrd)

output: 2D .png slices from the labels

Takes a single label or a directory and deconstructs the 3D scan into 2D slices of the specified size.

```
usage: labels_preprocess.py [-h] (--image IMAGE | --dir DIR)
                            [--desired_width DESIRED_WIDTH]
                            [--desired_height DESIRED_HEIGHT] --out OUT

Label pre-processing

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --image IMAGE         Input 3D label (default: None)
  --dir DIR             Input directory with 3D labels (default: None)

Resizing parameters:
  --desired_width DESIRED_WIDTH
                        width of the images (default: 512)
  --desired_height DESIRED_HEIGHT
                        height of the images (default: 512)

Output parameters:
  --out OUT             Output directory of the label slices (default: None)
```

### Training

#### Data augmentation

The data augmentation used in the training applies random rotation, shift, shear and zoom.

```
python3 src/py/heat_map.py --help
```

input: Database that contains the labels

output: Save the the heat map

Visualisation of the data augmentation that is applied on your dataset

```
usage: heat_map.py [-h] --dir_database DIR_DATABASE [--width WIDTH]
                   [--height HEIGHT] --out OUT

Visualization of the data augmentation

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir_database DIR_DATABASE
                        Input dir of the dataset (default: None)

label parameters:
  --width WIDTH         width of the images (default: 512)
  --height HEIGHT       height of the images (default: 512)

Output parameters:
  --out OUT             Output file (default: None)
```

#### Training  model

The neural network choosen is a U-Net architecture. The loss function is the *BinaryCrossentropy*. The metrics monitered during the training were the *Recall* and *Precision* metrics. 

```
python3 src/py/training_seg.py --help
```

input: The scans and labels for the training

output: The trained models, saved every *save_frequence* epochs

The algorithm takes as an input the training fold. It will use one of the cross validation folder as a validation set and the rest as a training set. The *save_frequence* parameter allow you to save the model at specific epochs. 

```
usage: training_seg.py [-h] --dir_train DIR_TRAIN --save_model SAVE_MODEL
                       --log_dir LOG_DIR
                       (--val_folds VAL_FOLDS [VAL_FOLDS ...] | --val_dir VAL_DIR)
                       [--model_name MODEL_NAME] [--epochs EPOCHS]
                       [--ratio RATIO] [--save_frequence SAVE_FREQUENCE]
                       [--learning_rate_schedular LEARNING_RATE_SCHEDULAR]
                       [--width WIDTH] [--height HEIGHT]
                       [--batch_size BATCH_SIZE]
                       [--learning_rate LEARNING_RATE]
                       [--number_filters NUMBER_FILTERS] [--dropout DROPOUT]

Training a neural network

optional arguments:
  -h, --help            show this help message and exit
  --val_folds VAL_FOLDS [VAL_FOLDS ...]
                        Fold of the cross-validation to keep for validation
                        (default: None)
  --val_dir VAL_DIR     Directory for the validation dataset (default: None)

Input files:
  --dir_train DIR_TRAIN
                        Input training folder (default: None)
  --save_model SAVE_MODEL
                        Directory to save the model (default: None)
  --log_dir LOG_DIR     Directory for the logs of the model (default: None)

training parameters:
  --model_name MODEL_NAME
                        Name of the model (default: CBCT_seg_model)
  --epochs EPOCHS       Number of epochs (default: 100)
  --ratio RATIO         Ratio of slices outside of the region of interest to
                        remove (between 0 and 1) (default: 0)
  --save_frequence SAVE_FREQUENCE
                        Epoch frequence to save the model (default: 5)
  --learning_rate_schedular LEARNING_RATE_SCHEDULAR
                        Set the LRS (default: None)
  --width WIDTH         width of the images (default: 512)
  --height HEIGHT       height of the images (default: 512)
  --batch_size BATCH_SIZE
                        Batch size value (default: 32)
  --learning_rate LEARNING_RATE
                        Learning rate (default: 0.0001)
  --number_filters NUMBER_FILTERS
                        Number of filters (default: 32)
  --dropout DROPOUT     Dropout (default: 0.1)
```

### Prediction

```
python3 src/py/predict_seg.py --help
```

input: 2D slices form CBCT scans

output: 2D slices predicted by the model

Takes 2D slices as an input from a CBCT scan and outputs the segmentation predicted for each slices. The output is not binary, it is a probability of each pixel to belong to the segmentation (value between [0;255]).

```
usage: predict_seg.py [-h] --dir_predict DIR_PREDICT [--width WIDTH]
                      [--height HEIGHT] --load_model LOAD_MODEL --out OUT

Prediction

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir_predict DIR_PREDICT
                        Input dir to be predicted (default: None)

Predict parameters:
  --width WIDTH         width of the images (default: 512)
  --height HEIGHT       height of the images (default: 512)
  --load_model LOAD_MODEL
                        Path of the trained model (default: None)

Output parameters:
  --out OUT             Output directory (default: None)
```

### Post-Processing

```
python3 src/py/postprocess.py --help
```

input: directory with .png slices

output: directory with reconstructed 3D image

Takes an input directory containing .png slices and reconstructs the 3D image. Post-processing is applied to improve the segmentation, depending on the tool_name (RCSeg or MandSeg). The output is saved with the same extension than the original scan.

```
usage: postprocess.py [-h] --dir DIR --original_dir ORIGINAL_DIR [--tool TOOL]
                      [--threshold THRESHOLD] --out OUT [--out_raw OUT_RAW]

Post-processing

optional arguments:
  -h, --help            show this help message and exit

Input files:
  --dir DIR             Input directory with 2D images (default: None)
  --original_dir ORIGINAL_DIR
                        Input directory with original 3D images (default:
                        None)

Parameters:
  --tool TOOL           Name of the tool used (default: MandSeg)
  --threshold THRESHOLD
                        if -1, the thresold apply is otsu, else it is the
                        value entered (between [0;255]) (default: -1)

Output parameters:
  --out OUT             Output directory (default: None)
  --out_raw OUT_RAW     Output directory for raw files (default: None)
```

### Evaluation of the trained models

```
python3 src/py/metrics.py --help
```

input: either a 3D file and its ground truth, or a directory of 3D files and their ground truths.

output: excel file containing the mean evaluation metrics (AUPRC, AUPRC - Baseline, F1-score, F2-score, Accuracy, Sensitivity, Precision) of the input files.

Compares the prediction made by the algorithm to the ground truth to evaluate the performances of the trained model.
If the output excel file already exist, it adds the results into a new line.

```
usage: metrics.py [-h] (--pred_img PRED_IMG | --pred_dir PRED_DIR)
                  [--pred_raw_img PRED_RAW_IMG | --pred_raw_dir PRED_RAW_DIR]
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
  --pred_raw_img PRED_RAW_IMG
                        Input raw predicted reconstructed 3D image (default:
                        None)
  --pred_raw_dir PRED_RAW_DIR
                        Input directory with raw predicted reconstructed 3D
                        images (default: None)
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
                        Learning rate (default: 1e-05)
  --number_filters NUMBER_FILTERS
                        Number of filters (default: 16)
  --cv_fold CV_FOLD     number of the cross-validation fold (default: 1)
```

