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
echo "--dir_metrics             Folder to save the metrics excel files of the evaluation of the model(s)"
echo "--cv_folds                Number of folds for the cross validation."
echo "--testing_percentage      Percentage of images to keep for testing"
echo "--min_percentage          Min percentage to threshold images for preprocessing"
echo "--max_percentage          Max percentage to threshold images for preprocessing"
echo "--model_name              Name of the model"
echo "--epochs                  Number of epochs for training the models"
echo "--ratio                   Ratio of slices outside of the region of interest to remove (value between [0;1])"
echo "--save_frequence          Frequence of saving the models"
echo "--width                   Width of the images"
echo "--height                  Height of the images"
echo "--learning_rate           Learning rate"
echo "--batch_size              Batch size"
echo "--NumberFilters           Number of filters"
echo "--dropout                 Dropout"
echo "--num_epoch               Number of the epoch of the model to select for the prediction"
echo "--tool_name               Name of the tool used"
echo "--threshold               Threshold to use to binarize scans in postprocess (-1 for otsu | [0;255] for a specific value)"
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
        --dir_metrics )  shift
            dir_metrics=$1;;
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
        --ratio )  shift
            ratio=$1;;
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
        --NumberFilters )  shift
            NumberFilters=$1;;
        --dropout )  shift
            dropout=$1;;
        --num_epoch )  shift
            num_epoch=$1;;
        --tool_name )  shift
            tool_name=$1;;
        --threshold )  shift
            threshold=$1;;
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
dir_src="${dir_src:-$dir_project/scripts/CBCT_seg}"
dir_data="${dir_data:-$dir_project/data}"
dir_database="${dir_database:-$dir_data/database}"
dir_cv="${dir_cv:-$dir_data/cross_validation}"
dir_test="$dir_cv/testing"
dir_train="$dir_cv/training"
dir_test_preproc="${dir_test_preproc:-$dir_cv/testing_preprocessed}"
dir_train_preproc="${dir_train_preproc:-$dir_cv/training_preprocessed}"
dir_test_predict="${dir_test_predict:-$dir_cv/testing_predicted}"
dir_train_predict="${dir_train_predict:-$dir_cv/training_predicted}"
dir_test_postproc="${dir_test_postproc:-$dir_cv/testing_postprocessed}"
dir_train_postproc="${dir_train_postproc:-$dir_cv/training_postprocessed}"
dir_metrics="${dir_metrics:-$dir_data/metrics}"

model_name="${model_name:-MandSeg_model}"
dir_model="${dir_model:-$dir_project/models/$model_name}"
dir_log="${dir_log:-$dir_model/log_dir}"

cv_folds="${cv_folds:-8}"
testing_percentage="${testing_percentage:-15}"
min_percentage="${min_percentage:-30}"
max_percentage="${max_percentage:-90}"
epochs="${epochs:-80}"
ratio="${ratio:-0.5}"
save_frequence="${save_frequence:-5}"
width="${width:-256}"
height="${height:-256}"
learning_rate="${learning_rate:-0.0001}"
batch_size="${batch_size:-32}"
NumberFilters="${NumberFilters:-16}"
dropout="${dropout:-0.1}"
num_epoch="${num_epoch:-40}"
tool_name="${tool_name:-MandSeg}"
threshold="${threshold:--1}"

out_metrics_val="${out_metrics_val:-$dir_data/$dir_metrics/metrics_validation.xlsx}"
out_metrics_testing="${out_metrics_testing:-$dir_data/$dir_metrics/metrics_testing.xlsx}"

python3 $dir_src/src/py/generate_workspace.py \
        --dir $dir_database \
        --out $dir_cv \
        --cv_folds $cv_folds \
        --testing_percentage $testing_percentage \

folds=$(eval echo $dir_train/{1..$cv_folds})
for dir in $folds $dir_test
do
    outdir=$(echo $dir | sed -e "s|${dir_test}|${dir_test_preproc}|g" -e "s|${dir_train}|${dir_train_preproc}|g")
    python3 $dir_src/src/py/preprocess.py \
            --dir $dir/Scans \
            --desired_width $width \
            --desired_height $height \
            --min_percentage $min_percentage \
            --max_percentage $max_percentage \
            --out $outdir/Scans \

    python3 $dir_src/src/py/labels_preprocess.py \
            --dir $dir/Segs \
            --desired_width $width \
            --desired_height $height \
            --out $outdir/Segs \
done

for cv_fold in $(eval echo {1..$cv_folds})
do
    echo $cv_fold
    python3 $dir_src/src/py/training_seg.py \
            --dir_train $dir_train_preproc \
            --val_folds $cv_fold \
            --save_model $dir_model \
            --log_dir $dir_log \
            --model_name $model_name"_"$cv_fold \
            --epochs $epochs\
            --ratio $ratio \
            --save_frequence $save_frequence \
            --learning_rate_schedular True \
            --width $width \
            --height $height \
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --number_filters $NumberFilters \
            --dropout $dropout \
done

folds=$(eval echo $dir_train_preproc/{1..$cv_folds})
for dir in $folds
do
    dir_predict=$(echo $dir | sed -e "s|${dir_train_preproc}|${dir_train_predict}|g")
    dir_postproc=$(echo $dir | sed -e "s|${dir_train_preproc}|${dir_train_postproc}|g")
    dir_gt=$(echo $dir | sed -e "s|${dir_train_preproc}|${dir_train}|g")
    python3 $dir_src/src/py/predict_seg.py \
            --dir_predict $dir/Scans \
            --load_model $dir_model/$model_name"_"$(basename ${dir})"_"$num_epoch.hdf5 \
            --width $width \
            --height $height \
            --out $dir_predict \
    
    python3 $dir_src/src/py/postprocess.py \
            --dir $dir_predict \
            --original_dir $dir_gt/Scans \
            --tool $tool_name \
            --threshold $threshold \
            --out $dir_postproc \
            --out_raw $dir_postproc"_raw" \

    python3 $dir_src/src/py/metrics.py \
            --pred_dir $dir_postproc \
            --pred_raw_dir $dir_postproc"_raw" \
            --groundtruth_dir $dir_gt/Segs \
            --out $out_metrics_val \
            --tool $tool_name \
            --model_name $model_name \
            --epochs $num_epoch\
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --number_filters $NumberFilters \
            --cv_fold $(basename ${dir}) \
done

folds=$(eval echo $dir_test_preproc/{1..$cv_folds})
for dir in $folds
do
    dir_predict=$(echo $dir | sed -e "s|${dir_test_preproc}|${dir_test_predict}|g")
    dir_postproc=$(echo $dir | sed -e "s|${dir_test_preproc}|${dir_test_postproc}|g")
    python3 $dir_src/src/py/predict_seg.py \
            --dir_predict $(dirname ${dir})/Scans \
            --load_model $dir_model/$model_name"_"$(basename ${dir})"_"$num_epoch.hdf5 \
            --width $width \
            --height $height \
            --out $dir_predict \
    
    python3 $dir_src/src/py/postprocess.py \
            --dir $dir_predict \
            --original_dir $dir_test/Scans \
            --tool $tool_name \
            --threshold $threshold \
            --out $dir_postproc \
            --out_raw $dir_postproc"_raw" \

    python3 $dir_src/src/py/metrics.py \
            --pred_dir $dir_postproc \
            --pred_raw_dir $dir_postproc"_raw" \
            --groundtruth_dir $dir_test/Segs \
            --out $out_metrics_testing \
            --tool $tool_name \
            --model_name $model_name \
            --epochs $num_epoch\
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --number_filters $NumberFilters \
            --cv_fold $(basename ${dir}) \
done
