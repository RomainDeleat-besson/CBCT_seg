# CBCT_seg

Contributors : Celia Le, Romain Deleat-Besson

Scripts for MandSeg and RootCanalSeg projects

## Prerequisites

python 3.7.9 with the librairies: //add librairies

## What is it?

CBCT_seg is a tool for CBCT segmentation based on a machine learning approach.

The Convolutional Neural Network (CNN) used is a 2.5D U-Net.

It takes several CBCT scan in input, with the extensions: .nrrd, .nii, .gipl (with .gz or not)

## Running the code

main script 

### Creation of the folds for the cross validation

python3 src/py/CV_folds.py 

input: folder containing all the scans and segmentations

output: folder containing the scans and segmentations divided into training/testing. The training folder is also divided into the number of folds wanted.

Takes a folder, searches into all the subfolders and seperates the scans ans the segmentations. Moves thoses files into the output folder. A training folder and a testing folder are created (the training folder is devided into the specified number of folds). The percentage or number of files for testing is selected randomly inside each folder, according to the propotion of files in each of them, to prevent class imbalance.

```
usage: CV_folds.py [-h] --dir DIR --out OUT [--cv_folds CV_FOLDS]
                   [--testing_number TESTING_NUMBER | --testing_percentage TESTING_PERCENTAGE]

Creation of the cross-validation folders

optional arguments:
  -h, --help            show this help message and exit

Input file:
  --dir DIR             Input directory with 3D images (default: None)

Output parameters:
  --out OUT             Output directory (default: None)
  --cv_folds CV_FOLDS   Number of folds to create (default: 10)
  --testing_number TESTING_NUMBER
                        Number of scans to keep for testing (default: None)
  --testing_percentage TESTING_PERCENTAGE
                        Percentage of scans to keep for testing (default: 20)
```

### Pre-Processing

python3 src/py/PreProcess.py 

input: 3D CBCT scans (.nrrd, .nii, .gipl (with .gz or not))

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

Input file:
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

### Training

python3 src/py/training_Seg.py 

input: 

output: 

explainations

```
help
```

### Prediction

python3 src/py/predict_Seg.py 

input: 

output: 

explainations

```
help
```

### Post-Processing

python3 src/py/PostProcess.py 

input: directory with .png slices

output: directory with reconstructed 3D image

Takes an input directory containing .png slices and reconstructs the 3D image. Saves it as .nrrd.
Post-processing will be added to this function.

```
usage: PostProcess.py [-h] --dir DIR --original_dir ORIGINAL_DIR --out OUT

Post-processing

optional arguments:
  -h, --help            show this help message and exit

Input file:
  --dir DIR             Input directory with 2D images (default: None)
  --original_dir ORIGINAL_DIR
                        Input directory with original 3D images (default:
                        None)

Output parameters:
  --out OUT             Output directory (default: None)
```

### Evaluation of the trained models

python3 src/py/metrics.py 

input: either a 3D file and its ground truth, or a directory of 3D files and their ground truths.

output: excel file containing the mean evaluation metrics (AUC, F1 score, Accuracy, Sensitivity, Precision) of the input files.

Compares the prediction made by the algorithm to the ground truth to evaluate the performances of the trained model.
If the output excel file already exist, it adds the results into a new line.

```
usage: metrics.py [-h] (--pred_img PRED_IMG | --pred_dir PRED_DIR)
                  (--groundtruth_img GROUNDTRUTH_IMG | --groundtruth_dir GROUNDTRUTH_DIR)
                  --out OUT [--sheet_name SHEET_NAME]
                  [--model_name MODEL_NAME] [--epochs EPOCHS]
                  [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                  [--number_filters NUMBER_FILTERS]
                  [--neighborhood {1,3,5,7,9}] [--cv_fold CV_FOLD]

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
  --sheet_name SHEET_NAME
                        Name of the excel sheet to write on (default: Sheet1)

Universal ID parameters:
  --model_name MODEL_NAME
                        name of the model (default: CBCT_seg_model)
  --epochs EPOCHS       name of the model (default: 20)
  --batch_size BATCH_SIZE
                        batch_size value (default: 32)
  --learning_rate LEARNING_RATE
  --number_filters NUMBER_FILTERS
  --neighborhood {1,3,5,7,9}
                        neighborhood slices (3|5|7) (default: 3)
  --cv_fold CV_FOLD     number of the cross-validation fold (default: 1)
```

