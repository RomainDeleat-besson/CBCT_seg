#!/bin/sh

Help()
{
# Display Help
echo "Program to train and evaluate a 2D U-Net segmentation model"
echo
echo "Syntax: main_training.sh [-i|d|o|s|m|seed1|seed_end|nbr_folds|h]"
echo "options:"
echo "--dir_project             Folder containing the project."
echo "--dir_src                 Folder containing the scripts."
echo "--dir_input               Folder containing the scans to segment."
echo "--dir_preproc             Folder to save the preprocessed images"
echo "--dir_model               Folder to save the models"
echo "--dir_predict             Folder to save the predicted images"
echo "--dir_postproc            Folder to save the postprocessed images"
echo "--cv_fold                 Number of the fold of the cross validation to select."
echo "--model_name              Name of the model"
echo "--width                   Width of the images"
echo "--height                  Height of the images"
echo "--neighborhood            Size of the neighborhood slices"
echo "--num_epoch               Number of the epoch of the model to select for the prediction"
echo "-h|--help                 Print this Help."
echo
}

while [ "$1" != "" ]; do
    case $1 in
        --dir_project )  shift
            dir_project=$1;;
        --dir_src )  shift
            dir_src=$1;;
        --dir_input )  shift
            dir_input=$1;;
        --dir_preproc )  shift
            dir_preproc=$1;;
        --dir_model )  shift
            dir_model=$1;;
        --dir_predict )  shift
            dir_predict=$1;;
        --dir_postproc )  shift
            dir_postproc=$1;;
        --cv_fold )  shift
            cv_fold=$1;;
        --model_name )  shift
            model_name=$1;;
        --width )  shift
            width=$1;;
        --height )  shift
            height=$1;;
        --neighborhood )  shift
            neighborhood=$1;;
        --num_epoch )  shift
            num_epoch=$1;;
        -h | --help )
            Help
            exit;;
        * ) 
            echo ' - Error: Unsupported flag'
            Help
            exit 1
    esac
    shift
done

dir_project="${dir_database:-/Users/luciacev-admin/Documents/MandSeg}"
dir_src="${dir_src:-$dir_project/CBCT_seg}"
dir_input="${dir_cv:-$dir_project/data/Scans}"
dir_preproc="${dir_preproc:-$dir_input"_PreProcessed"}"
dir_model="${dir_model:-$dir_project/models/$model_name}"
dir_predict="${dir_predict:-$dir_input"_Predicted"}"
dir_postproc="${dir_postproc:-$dir_input"_PostProcessed"}"

cv_fold="${cv_fold:-1}"
model_name="${model_name:-CBCT_seg_model}"
width="${width:-512}"
height="${height:-512}"
neighborhood="${neighborhood:-3}"
num_epoch="${num_epoch:-1}"


python3 CBCT_seg/src/py/PreProcess.py \
    --dir $dir_input \
    --out $dir_preproc 

python3 CBCT_seg/src/py/predict_Seg.py \
    --dir_predict $dir_preproc \
    --load_model $dir_model/$modelName/$cv_fold/$modelName"_"$num_epoch.hdf5 \
    --width $width \
    --height $height \
    --neighborhood $neighborhood \
    --out $dir_predict

python3 CBCT_seg/src/py/PostProcess.py \
    --dir $dir_predict \
    --original_dir $dir_input \
    --out $dir_postproc

