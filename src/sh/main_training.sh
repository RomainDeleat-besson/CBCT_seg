#!/bin/sh

Help()
{
# Display Help
echo "Program to train and evaluate a 2D U-Net segmentation model"
echo
echo "Syntax: main_training.sh [--options]"
echo "options:"
echo "--dir_project             Folder containing the project."
echo "--dir_src                 Folder containing the scripts."
echo "--dir_data                Folder containing all the data of the project."
echo "--dir_database            Folder containing the database for the training/testing."
echo "--dir_cv                  Folder to save the cross-validation folds."
echo "--dir_test_preproc        Folder to save the preprocessed testing images"
echo "--dir_train_preproc       Folder to save the preprocessed training images"
echo "--dir_model               Folder to save the models"
echo "--dir_log                 Folder to save the logs of the model"
echo "--dir_test_predict        Folder to save the predicted testing images"
echo "--dir_train_predict       Folder to save the predicted training images"
echo "--dir_test_postproc       Folder to save the postprocessed testing images"
echo "--dir_train_postproc      Folder to save the postprocessed training images"
echo "--cv_folds                Number of folds for the cross validation."
echo "--testing_percentage      Percentage of images to keep for testing"
echo "--min_percentage          Min percentage to threshold images for preprocessing"
echo "--max_percentage          Max percentage to threshold images for preprocessing"
echo "--model_name              Name of the model"
echo "--epochs                  Number of epochs for training the models"
echo "--save_frequence          Frequence of saving the models"
echo "--width                   Width of the images"
echo "--height                  Height of the images"
echo "--learning_rate           Learning rate"
echo "--batch_size              Batch size"
echo "--neighborhood            Size of the neighborhood slices"
echo "--NumberFilters           Number of filters"
echo "--dropout                 Dropout"
echo "--num_epoch               Number of the epoch of the model to select for the prediction"
echo "--out_metrics_val         File to save the evaluation metrics of the models on the validation set"
echo "--out_metrics_testing     File to save the evaluation metrics of the models on the testing set"
echo "-h|--help                 Print this Help."
echo
}

while [ "$1" != "" ]; do
    case $1 in
        --dir_project )  shift
            dir_project=$1;;
        --dir_src )  shift
            dir_src=$1;;
        --dir_data )  shift
            dir_data=$1;;
        --dir_database )  shift
            dir_database=$1;;
        --dir_cv )  shift
            dir_cv=$1;;
        --dir_test_preproc )  shift
            dir_test_preproc=$1;;
        --dir_train_preproc )  shift
            dir_train_preproc=$1;;
        --dir_model )  shift
            dir_model=$1;;
        --dir_log )  shift
            dir_log=$1;;
        --dir_test_predict )  shift
            dir_test_predict=$1;;
        --dir_train_predict )  shift
            dir_train_predict=$1;;
        --dir_test_postproc )  shift
            dir_test_postproc=$1;;
        --dir_train_postproc )  shift
            dir_train_postproc=$1;;
        --cv_folds )  shift
            cv_folds=$1;;
        --testing_percentage )  shift
            testing_percentage=$1;;
        --min_percentage )  shift
            min_percentage=$1;;
        --max_percentage )  shift
            max_percentage=$1;;
        --model_name )  shift
            model_name=$1;;
        --epochs )  shift
            epochs=$1;;
        --save_frequence )  shift
            save_frequence=$1;;
        --width )  shift
            width=$1;;
        --height )  shift
            height=$1;;
        --learning_rate )  shift
            learning_rate=$1;;
        --batch_size )  shift
            batch_size=$1;;
        --neighborhood )  shift
            neighborhood=$1;;
        --NumberFilters )  shift
            NumberFilters=$1;;
        --dropout )  shift
            dropout=$1;;
        --num_epoch )  shift
            num_epoch=$1;;
        --out_metrics_val )  shift
            out_metrics_val=$1;;
        --out_metrics_testing )  shift
            out_metrics_testing=$1;;
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

dir_project="${dir_project:-/Users/luciacev-admin/Documents/MandSeg}"
dir_src="${dir_src:-$dir_project/master/CBCT_seg}"
dir_data="${dir_data:-$dir_project/data}"
dir_database="${dir_database:-$dir_data/database}"
dir_cv="${dir_cv:-$dir_data/CV}"
dir_test="$dir_cv/Testing"
dir_train="$dir_cv/Training"
dir_test_preproc="${dir_test_preproc:-$dir_cv/Testing_PreProcessed}"
dir_train_preproc="${dir_train_preproc:-$dir_cv/Training_PreProcessed}"
dir_model="${dir_model:-$dir_project/models/$model_name}"
dir_log="${dir_log:-$dir_model/log_dir}"
dir_test_predict="${dir_test_predict:-$dir_cv/Testing_Predicted}"
dir_train_predict="${dir_train_predict:-$dir_cv/Training_Predicted}"
dir_test_postproc="${dir_test_postproc:-$dir_cv/Testing_PostProcessed}"
dir_train_postproc="${dir_train_postproc:-$dir_cv/Training_PostProcessed}"
cv_folds="${cv_folds:-10}"
testing_percentage="${testing_percentage:-20}"
min_percentage="${min_percentage:-45}"
max_percentage="${max_percentage:-90}"
model_name="${model_name:-CBCT_seg_model}"
epochs="${epochs:-50}"
save_frequence="${save_frequence:-5}"
width="${width:-512}"
height="${height:-512}"
learning_rate="${learning_rate:-0.0001}"
batch_size="${batch_size:-16}"
neighborhood="${neighborhood:-3}"
NumberFilters="${NumberFilters:-64}"
dropout="${dropout:-0.1}"
num_epoch="${num_epoch:-1}"

out_metrics_val="${out_metrics_val:-$dir_data/out/metrics_validation.xlsx}"
out_metrics_testing="${out_metrics_testing:-$dir_data/out/metrics_testing.xlsx}"

python3 $dir_src/src/py/CV_folds.py \
        --dir $dir_database \
        --out $dir_cv \
        --cv_folds $cv_folds \
        --testing_percentage $testing_percentage

folds=$(eval echo $dir_train/{1..$cv_folds})
for dir in $folds $dir_test
do
    outdir=$(echo $dir | sed -e "s|${dir_test}|${dir_test_preproc}|g" -e "s|${dir_train}|${dir_train_preproc}|g")
    python3 $dir_src/src/py/PreProcess.py \
            --dir $dir/Scans \
            --desired_width $width \
            --desired_height $height \
            --min_percentage $min_percentage \
            --max_percentage $max_percentage \
            --out $outdir/Scans 

    python3 $dir_src/src/py/labels_preprocess.py \
            --dir $dir/Segs \
            --out $outdir/Segs
done

for cv_fold in $(eval echo {1..$cv_folds})
do
    echo $cv_fold
    python3 $dir_src/src/py/training_Seg.py \
            --dir_train $dir_train \
            --val_folds $cv_fold \
            --save_model $dir_model \
            --log_dir $dir_log \
            --model_name $model_name/$cv_fold \
            --epochs $epochs\
            --save_frequence $save_frequence \
            --width $width \
            --height $height \
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --neighborhood $neighborhood \
            --number_filters $NumberFilters \
            --dropout $dropout
done

folds=$(eval echo $dir_train_preproc/{1..$cv_folds})
for dir in $folds
do
    dir_predict=$(echo $dir | sed -e "s|${dir_train_preproc}|${dir_train_predict}|g")
    dir_postproc=$(echo $dir | sed -e "s|${dir_train_preproc}|${dir_train_postproc}|g")
    dir_gt=$(echo $dir | sed -e "s|${dir_train_preproc}|${dir_train}|g")
    python3 $dir_src/src/py/predict_Seg.py \
            --dir_predict $dir \
            --load_model $dir_model/$model_name/$cv_fold/$model_name"_"$num_epoch.hdf5 \
            --width $width \
            --height $height \
            --neighborhood $neighborhood \
            --out $dir_predict

    python3 $dir_src/src/py/PostProcess.py \
            --dir $dir_predict \
            --original_dir $dir_gt/Segs \
            --out $dir_postproc

    python3 $dir_src/src/py/metrics.py \
            --pred_dir $dir_postproc \
            --groundtruth_dir $dir_gt/Segs \
            --out $out_metrics_val \
            --model_name $model_name \
            --epochs $epochs\
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --neighborhood $neighborhood \
            --number_filters $NumberFilters \
            --cv_fold $(basename${dir})

done

folds=$(eval echo $dir_test_preproc/{1..$cv_folds})
for dir in $folds
do
    dir_predict=$(echo $dir | sed -e "s|${dir_test_preproc}|${dir_test_predict}|g")
    dir_postproc=$(echo $dir | sed -e "s|${dir_test_preproc}|${dir_test_postproc}|g")
    python3 $dir_src/src/py/predict_Seg.py \
            --dir_predict $dir \
            --load_model $dir_model/$model_name/$cv_fold/$model_name"_"$num_epoch.hdf5 \
            --width $width \
            --height $height \
            --neighborhood $neighborhood \
            --out $dir_predict
    
    python3 $dir_src/src/py/PostProcess.py \
            --dir $dir_predict \
            --original_dir $dir_test/Segs \
            --out $dir_postproc

    python3 $dir_src/src/py/metrics.py \
            --pred_dir $dir_postproc \
            --groundtruth_dir $dir_test/Segs \
            --out $out_metrics_testing \
            --model_name $model_name \
            --epochs $epochs\
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --neighborhood $neighborhood \
            --number_filters $NumberFilters \
            --cv_fold $(basename ${dir})

done
