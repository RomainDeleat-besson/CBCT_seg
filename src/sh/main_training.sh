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
echo "--dir_database            Folder containing the database."
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
echo "--out_metrics             File to save the evaluation metrics of the models"
echo "--sheet_name_val          Name of the sheet to write the metrics for the validation"
echo "--sheet_name_test         Name of the sheet to write the metrics for the testing"
echo "-h|--help                 Print this Help."
echo
}

while [ "$1" != "" ]; do
    case $1 in
        --dir_project )  shift
            dir_project=$1;;
        --dir_src )  shift
            dir_src=$1;;
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
        --out_metrics )  shift
            out_metrics=$1;;
        --sheet_name_val )  shift
            sheet_name_val=$1;;
        --sheet_name_test )  shift
            sheet_name_test=$1;;
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
dir_database="${dir_database:-$dir_project/data/database}"
dir_cv="${dir_cv:-$dir_project/data/CV}"
dir_test="$dir_cv/Testing"
dir_train="$dir_cv/Training"
dir_test_preproc="${dir_preproc:-$dir_cv/Testing_PreProcessed}"
dir_train_preproc="${dir_preproc:-$dir_cv/Training_PreProcessed}"
dir_model="${dir_model:-$dir_project/models/$model_name}"
dir_log="${dir_log:-$dir_model/log_dir}"
dir_test_predict="${dir_predict:-$dir_cv/Testing_Predicted}"
dir_train_predict="${dir_predict:-$dir_cv/Training_Predicted}"
dir_test_postproc="${dir_postproc:-$dir_cv/Testing_PostProcessed}"
dir_train_postproc="${dir_postproc:-$dir_cv/Training_PostProcessed}"

cv_folds="${cv_folds:-10}"
testing_percentage="${testing_percentage:-20}"
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

out_metrics="${out_metrics:-$dir_project/out/metrics.xlsx}"
sheet_name_val="${sheet_name_val:-Validation}"
sheet_name_test="${sheet_name_val:-Testing}"


python3 CBCT_seg/src/py/CV_folds.py \
        --dir $dir_database \
        --out $dir_cv \
        --cv_folds $cv_folds \
        --testing_percentage $testing_percentage

folds=$(eval echo $dir_train/{1..$cv_folds})
for dir in $folds $dir_test
do
    outdir=$(echo $dir | sed -e "s|${dir_test}|${dir_test_preproc}|g" -e "s|${dir_train}|${dir_train_preproc}|g")
    python3 CBCT_seg/src/py/PreProcess.py \
            --dir $dir/Scans \
            --out $outdir/Scans 

    python3 CBCT_seg/src/py/labels_preprocess.py \
            --dir $dir/Segs \
            --out $outdir/Segs
done

for cv_fold in $(eval echo {1..$cv_folds})
do
    echo $cv_fold
    python3 CBCT_seg/src/py/training_Seg.py \
            --dir_train $dir_train \
            --cv_fold $cv_fold \
            --save_model $dir_model \
            --log_dir $dir_log \
            --model_name $modelName/$cv_fold \
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
    python3 CBCT_seg/src/py/predict_Seg.py \
            --dir_predict $dir \
            --load_model $dir_model/$modelName/$cv_fold/$modelName"_"$num_epoch.hdf5 \
            --width $width \
            --height $height \
            --neighborhood $neighborhood \
            --out_dir $dir_predict

    python3 CBCT_seg/src/py/PostProcess.py \
            --dir $dir_predict \
            --original_dir $dir_cv \
            --out $dir_postproc

    python3 CBCT_seg/src/py/metrics.py \
            --pred_dir $dir_postproc \
            --groundtruth_dir $dir_gt \
            --out $out_metrics \
            --sheet_name $sheet_name_val \
            --model_name $modelName \
            --epochs $epochs\
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --neighborhood $neighborhood \
            --number_filters $NumberFilters \
            --cv_folds $cv_folds

done

folds=$(eval echo $dir_test_preproc/{1..$cv_folds})
for dir in $folds
do
    dir_predict=$(echo $dir | sed -e "s|${dir_test_preproc}|${dir_test_predict}|g")
    dir_postproc=$(echo $dir | sed -e "s|${dir_test_preproc}|${dir_test_postproc}|g")
    python3 CBCT_seg/src/py/predict_Seg.py \
            --dir_predict $dir \
            --load_model $dir_model/$modelName/$cv_fold/$modelName"_"$num_epoch.hdf5 \
            --width $width \
            --height $height \
            --neighborhood $neighborhood \
            --out_dir $dir_predict

    python3 CBCT_seg/src/py/PostProcess.py \
            --dir $dir_predict \
            --original_dir $dir_cv \
            --out $dir_postproc

    python3 CBCT_seg/src/py/metrics.py \
            --pred_dir $dir_postproc \
            --groundtruth_dir $dir_test \
            --out $out_metrics \
            --sheet_name $sheet_name_test \
            --model_name $modelName \
            --epochs $epochs\
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --neighborhood $neighborhood \
            --number_filters $NumberFilters \
            --cv_folds $cv_folds

done
