import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
from numba import jit, prange
from sklearn import metrics

from utils import *


@jit(nopython=True, nogil=True, cache=True, parallel=True, fastmath=True)
def compute_tp_tn_fp_fn(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in prange(y_pred.size):
        tp += y_true[i] * y_pred[i]
        tn += (1-y_true[i]) * (1-y_pred[i])
        fp += (1-y_true[i]) * y_pred[i]
        fn += y_true[i] * (1-y_pred[i])
         
    return tp, tn, fp, fn

def compute_precision(tp, fp):
    return tp / (tp + fp)

def compute_recall(tp, fn):
    return tp / (tp + fn)

def compute_f1_score(precision, recall):
    try:
        return (2*precision*recall) / (precision + recall)
    except:
        return 0

def compute_fbeta_score(precision, recall, beta):
    try:
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    except:
        return 0

def compute_accuracy(tp,tn,fp,fn):
    return (tp + tn)/(tp + tn + fp + fn)

def compute_auc(GT, pred):
    return metrics.roc_auc_score(GT, pred)

def compute_auprc(GT, pred):
    prec, rec, thresholds = metrics.precision_recall_curve(GT, pred)
    return metrics.auc(prec, rec)

def compute_average_precision(GT, pred):
    return metrics.average_precision_score(GT, pred)
    

def main(args):
    #====== Numba compilation ======
    # The 2 lines are important
    compute_tp_tn_fp_fn(np.array([0,0,0], dtype=np.uint8), np.array([0,1,0], dtype=np.uint8))
    compute_tp_tn_fp_fn(np.array([0,0,0], dtype=np.float32), np.array([0,1,0], dtype=np.float32))
    #===============================

    out = args.out
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

    model_name = args.model_name
    number_epochs = args.epochs
    batch_size = args.batch_size
    NumberFilters = args.number_filters
    lr = args.learning_rate
    cv_fold = args.cv_fold
    model_params = ['Number Epochs', 'Batch Size', 'Number Filters', 'Learning Rate', 'Empty col', 'Empty col2', 'Empty col3', 'CV']
    param_values = [number_epochs, batch_size, NumberFilters, lr, '', '', '', '']
    Params = pd.Series(param_values, index=model_params, name='Params values')
    metrics_names = ['AUC','AUPRC','F1_Score','Fbeta_Score','Accuracy','Recall','Precision','CV fold']
    Metrics = pd.Series(metrics_names, index=model_params, name='Model\Metrics')

    if not os.path.exists(out): 
        Folder_Metrics = pd.DataFrame(columns = model_params)
        Image_Metrics = pd.DataFrame(columns = model_params)
    else: 
        Metrics_file = pd.ExcelFile(out)
        Folder_Metrics = pd.read_excel(Metrics_file, 'Sheet1', index_col=0, header=None)
        Folder_Metrics = Folder_Metrics[Folder_Metrics.columns[:8]]
        Folder_Metrics.columns = model_params
        Image_Metrics = pd.read_excel(Metrics_file, 'Sheet2', index_col=0, header=None)
        Image_Metrics.columns = model_params

    matching_values = (Folder_Metrics.values[:,:-2] == Params.values[:-2]).all(1)
    if not matching_values.any():
        Folder_Metrics = Folder_Metrics.append(pd.Series(['Number Epochs', 'Batch Size', 'Number Filters', 'Learning Rate', '', '', '', 'CV'], name='Params', index=model_params), ignore_index=False)
        Folder_Metrics = Folder_Metrics.append(Params, ignore_index=False)
        Folder_Metrics = Folder_Metrics.append(Metrics, ignore_index=False)
        Folder_Metrics = Folder_Metrics.append(pd.Series(name='', dtype='object'), ignore_index=False)

    matching_values = (Image_Metrics.values[:,:-2] == Params.values[:-2]).all(1)
    if not matching_values.any():
        Image_Metrics = Image_Metrics.append(pd.Series(['Number Epochs', 'Batch Size', 'Number Filters', 'Learning Rate', '', '', '', 'File Name'], name='Params', index=model_params), ignore_index=False)
        Image_Metrics = Image_Metrics.append(pd.Series(param_values, index=model_params, name='Params values'), ignore_index=False)
        Image_Metrics = Image_Metrics.append(pd.Series(['AUC','AUPRC','F1_Score','Fbeta_Score','Accuracy','Recall','Precision','File Name'], index=model_params, name='Model\Metrics'), ignore_index=False)
        Image_Metrics = Image_Metrics.append(pd.Series(name='', dtype='object'), ignore_index=False)

    arrays = [range(len(Folder_Metrics)), Folder_Metrics.index]
    Index = pd.MultiIndex.from_arrays(arrays, names=('number', 'name'))
    Folder_Metrics.set_index(Index, inplace=True)
    arrays = [range(len(Image_Metrics)), Image_Metrics.index]
    Index = pd.MultiIndex.from_arrays(arrays, names=('number', 'name'))
    Image_Metrics.set_index(Index, inplace=True)
    idx1 = Folder_Metrics[(Folder_Metrics.values[:,:-2] == Params.values[:-2]).all(1)].index.get_level_values('number').tolist()[0]
    idx2 = Image_Metrics[(Image_Metrics.values[:,:-2] == Params.values[:-2]).all(1)].index.get_level_values('number').tolist()[0]
    img_fn_array = []

    if args.pred_img:
        img_obj = {}
        img_obj["img"] = args.pred_img
        img_obj["GT"] = args.groundtruth_img
        img_fn_array.append(img_obj)

    if args.pred_dir:
        normpath_img = os.path.normpath("/".join([args.pred_dir, '*', '']))
        normpath_GT = os.path.normpath("/".join([args.groundtruth_dir, '*', '']))

        img_list = []
        for img_fn in glob.iglob(normpath_img, recursive=True):
            if args.tool == 'RCSeg':
                img_split = os.path.basename(img_fn).split("_")
                if img_split[0] == img_split[-2] or (img_split[-2] not in ['upper', 'lower']):
                    img_list.append(img_fn)
            else: 
                img_list.append(img_fn)

        for (img_fn, GT_fn) in zip(sorted(img_list), sorted(glob.iglob(normpath_GT, recursive=True))):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                img_obj = {}
                img_obj["img"] = img_fn
                img_obj["GT"] = GT_fn
                img_fn_array.append(img_obj)

    total_values = pd.DataFrame(columns=model_params)
    
    for img_obj in img_fn_array:
        startTime = time.time()
        pred_path = img_obj["img"]
        GT_path = img_obj["GT"]

        pred, _ = ReadFile(pred_path)
        GT, _ = ReadFile(GT_path, verbose=0)

        pred = Normalize(pred,out_min=0,out_max=1)
        GT = Normalize(GT,out_min=0,out_max=1)
        pred[pred<=0.5]=0
        pred[pred>0.5]=1
        GT[GT<=0.5]=0
        GT[GT>0.5]=1

        pred = np.array(pred).flatten()

        GT = np.array(GT).flatten()
        GT = np.uint8(GT > 0.5)

        tp, tn, fp, fn = compute_tp_tn_fp_fn(GT, pred)
        recall = compute_recall(tp, fn)
        precision = compute_precision(tp, fp)
        f1 = compute_f1_score(precision, recall)
        fbeta = compute_fbeta_score(precision, recall, 2)
        acc = compute_accuracy(tp, tn, fp, fn)
        auc = compute_auc(GT, pred)
        # auprc = compute_auprc(GT, pred)
        auprc = compute_average_precision(GT, pred)

        metrics_line = [auc,auprc,f1,fbeta,acc,recall,precision]
        metrics_line.append(os.path.basename(pred_path).split('.')[0])
        total_values.loc[len(total_values)] = metrics_line

        stopTime = time.time()
        print('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))


    means = total_values[total_values.columns.drop('CV')].mean()
    stds = total_values[total_values.columns.drop('CV')].std()
    stds = [0 if math.isnan(x) else x for x in stds]
    values = [(f"{mean:.4f}"+' \u00B1 '+f"{std:.4f}") for (mean,std) in zip(means,stds)]
    values.append(cv_fold)
    line = pd.DataFrame([values], columns=model_params)
    Index_line = pd.MultiIndex.from_arrays([[idx1+1.5],[model_name]], names=('number', 'name'))
    line.set_index(Index_line, inplace=True)
    Folder_Metrics = Folder_Metrics.append(line, ignore_index=False)
    Folder_Metrics = Folder_Metrics.sort_index()
    Folder_Metrics = Folder_Metrics.set_index(Folder_Metrics.index.droplevel('number').rename('Params'))

    index_number = [idx2+1+(1/(len(total_values)+1)*(i+1)) for i in range(len(total_values))]
    index_name = [model_name for i in range(len(total_values))]
    Index_line = pd.MultiIndex.from_arrays([index_number,index_name], names=('number', 'name'))
    total_values.set_index(Index_line, inplace=True)
    Image_Metrics = Image_Metrics.append(total_values, ignore_index=False)
    Image_Metrics = Image_Metrics.sort_index()
    Image_Metrics = Image_Metrics.set_index(Image_Metrics.index.droplevel('number').rename('Params'))

    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    Folder_Metrics.to_excel(writer, sheet_name='Sheet1', header=False)
    Image_Metrics.to_excel(writer, sheet_name='Sheet2', header=False)
    workbook = writer.book
    worksheet1 = writer.sheets['Sheet1']
    worksheet2 = writer.sheets['Sheet2']

    row_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
    for ind, row in enumerate(Folder_Metrics.index):
        if row in ['Params', 'Model\Metrics']:
            worksheet1.set_row(ind, 15, row_format)
    for ind, row in enumerate(Image_Metrics.index):
        if row in ['Params', 'Model\Metrics']:
            worksheet2.set_row(ind, 15, row_format)
        elif row not in ['Params values']:
            worksheet2.set_row(ind, 15, workbook.add_format({'num_format': '0.0000', 'align': 'center', 'valign': 'vcenter'}))

    col_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    for ind, col in enumerate(Folder_Metrics.columns):
        column_len = Folder_Metrics[col].astype(str).str.len().max() + 2
        worksheet1.set_column(ind+1, ind+1, column_len, col_format)
    for ind, col in enumerate(Image_Metrics.columns):
        column_len = Image_Metrics[col].astype(str).str.len().max() + 2
        worksheet2.set_column(ind+1, ind+1, column_len, col_format)

    indexcol_len = Folder_Metrics.index.astype(str).str.len().max() + 2
    worksheet1.set_column(0, 0, indexcol_len, col_format)
    indexcol_len = Image_Metrics.index.astype(str).str.len().max() + 2
    worksheet2.set_column(0, 0, indexcol_len, col_format)

    writer.save()



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Evaluation metrics', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_params = parser.add_argument_group('Input files')
    predicted_files = input_params.add_mutually_exclusive_group(required=True)
    predicted_files.add_argument('--pred_img', type=str, help='Input predicted reconstructed 3D image')
    predicted_files.add_argument('--pred_dir', type=str, help='Input directory with predicted reconstructed 3D images')

    groundtruth_files = input_params.add_mutually_exclusive_group(required=True)
    groundtruth_files.add_argument('--groundtruth_img', type=str, help='Input original 3D images (ground truth)')
    groundtruth_files.add_argument('--groundtruth_dir', type=str, help='Input directory with original 3D images (ground truth)')

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output filename', required=True)

    training_parameters = parser.add_argument_group('Training parameters')
    training_parameters.add_argument('--tool', type=str, help='Name of the tool used', default='MandSeg')
    training_parameters.add_argument('--model_name', type=str, help='name of the model', default='CBCT_seg_model')
    training_parameters.add_argument('--epochs', type=int, help='name of the model', default=20)
    training_parameters.add_argument('--batch_size', type=int, help='batch_size value', default=16)
    training_parameters.add_argument('--learning_rate', type=float, help='', default=0.00001)
    training_parameters.add_argument('--number_filters', type=int, help='', default=16)
    training_parameters.add_argument('--cv_fold', type=int, help='number of the cross-validation fold', default=1)

    args = parser.parse_args()

    main(args)
