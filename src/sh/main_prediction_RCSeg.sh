#!/bin/sh

Help()
{
# Display Help
echo "Program to train and evaluate a 2D U-Net segmentation model"
echo
echo "Syntax: main_prediction_RCSeg.sh [--options]"
echo "options:"
echo "--dir_src                 Path to the Folder that contains the source code"
echo "--file_input              Scan to segment"
echo "--dir_preproc             Folder to save the preprocessed images"
echo "--dir_predicted           Folder to save the predicted images"
echo "--dir_output              Folder to save the postprocessed images"
echo "--width                   Width of the images"
echo "--height                  Height of the images"
echo "--tool_name               Tool name [MandSeg | RCSeg]"
echo "--threshold               Threshold to use to binarize scans in postprocess. (-1 for otsu | [0;255] for a specific value)"
echo "-h|--help                 Print this Help"
echo
}

while [ "$1" != "" ]; do
    case $1 in
        --dir_src )  shift
            dir_src=$1;;
        --file_input )  shift
            file_input=$1;;
        --dir_preproc )  shift
            dir_preproc=$1;;
        --dir_predicted )  shift
            dir_predicted=$1;;
        --dir_output )  shift
            dir_output=$1;;
        --path_model )  shift
            path_model=$1;;
        --min_percentage )  shift
            min_percentage=$1;;
        --max_percentage )  shift
            max_percentage=$1;;
        --width )  shift
            width=$1;;
        --height )  shift
            height=$1;;
        --tool_name )  shift
            tool_name=$1;;
        --threshold )  shift
            threshold=$1;;
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

dir_src="${dir_src:-./CBCT_seg/src}"
# dir_input="${dir_input:-./Scans}"
dir_preproc="${dir_preproc:-/app/data/preproc}"
dir_predicted="${dir_predicted:-/app/data/predicted}"
dir_output="${dir_output:-$(dirname $file_input)}"

min_percentage="${min_percentage:-55}"
max_percentage="${max_percentage:-90}"
width="${width:-512}"
height="${height:-512}"
tool_name="${tool_name:-RCSeg}"
threshold="${threshold:-100}"


python3 $dir_src/py/preprocess.py \
    --image $file_input \
    --desired_width $width \
    --desired_height $height \
    --min_percentage $min_percentage \
    --max_percentage $max_percentage \
    --out $dir_preproc \

python3 $dir_src/py/predict_seg.py \
    --dir_predict $dir_preproc \
    --load_model $path_model \
    --width $width \
    --height $height \
    --out $dir_predicted \

python3 $dir_src/py/postprocess.py \
    --dir $dir_predicted \
    --original_dir $(dirname $file_input) \
    --tool $tool_name \
    --threshold $threshold \
    --out $dir_output \

