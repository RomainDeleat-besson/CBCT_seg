import argparse
import glob
import os

import itk
import numpy as np
import pandas as pd
from sklearn import metrics

from utils import *


def main(args):

    out = args.out
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))

    sheet = args.sheet_name

    model_name = args.model_name
    number_epochs = args.epochs
    neighborhood = args.neighborhood
    batch_size = args.batch_size
    NumberFilters = args.number_filters
    lr = args.learning_rate
    cv_fold = args.cv_fold
    model_params = ['Number Epochs', 'Neighborhood', 'Batch Size', 'Number Filters', 'Learning Rate', 'CV']
    param_values = [number_epochs, neighborhood, batch_size, NumberFilters, lr, '']
    Params = pd.Series(param_values, index=model_params, name='Params values')
    metrics_names = ['AUC','F1_Score','Accuracy','Sensitivity','Precision','CV fold']
    Metrics = pd.Series(metrics_names, index=model_params, name='Model\Metrics')

    if not os.path.exists(out): 
        Excel_Metrics = pd.DataFrame(columns = model_params)
    else: 
        Excel_Metrics = pd.read_excel(out, index_col=0, header=None)
        Excel_Metrics.columns = model_params

    matching_values = (Excel_Metrics.values[:,:-1] == Params.values[:-1]).all(1)
    if not matching_values.any():
        Excel_Metrics = Excel_Metrics.append(pd.Series(model_params, name='Params', index=model_params), ignore_index=False)
        Excel_Metrics = Excel_Metrics.append(Params, ignore_index=False)
        Excel_Metrics = Excel_Metrics.append(Metrics, ignore_index=False)
        Excel_Metrics = Excel_Metrics.append(pd.Series(name='', dtype='object'), ignore_index=False)

    arrays = [range(len(Excel_Metrics)), Excel_Metrics.index]
    Index = pd.MultiIndex.from_arrays(arrays, names=('number', 'name'))
    Excel_Metrics.set_index(Index, inplace=True)

    # print(Excel_Metrics)
    idx_nbr = Excel_Metrics[(Excel_Metrics.values[:,:-1] == Params.values[:-1]).all(1)].index.get_level_values('number').tolist()[0]

    img_fn_array = []

    if args.pred_img:
        img_obj = {}
        img_obj["pred"] = args.pred_img
        img_obj["GT"] = args.groundtruth_img
        img_fn_array.append(img_obj)

    if args.pred_dir:
        normpath_img = os.path.normpath("/".join([args.pred_dir, '*', '']))
        normpath_GT = os.path.normpath("/".join([args.groundtruth_dir, '*', '']))
        for (img_fn, GT_fn) in zip(sorted(glob.iglob(normpath_img, recursive=True)), sorted(glob.iglob(normpath_GT, recursive=True))):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                img_obj = {}
                img_obj["img"] = img_fn
                img_obj["GT"] = GT_fn
                img_fn_array.append(img_obj)

    auc = f1 = acc = sensitivity = precision = []

    for img_obj in img_fn_array:
        pred = img_obj["img"]
        GT = img_obj["GT"]

        pred, _ = ReadFile(pred)
        GT, _ = ReadFile(GT, verbose=0)

        # pred = Normalize(pred,out_min=0,out_max=1)
        # GT = Normalize(GT,out_min=0,out_max=1)
        # pred[pred<=0.5]=0
        # pred[pred>0.5]=1
        # GT[GT<=0.5]=0
        # GT[GT>0.5]=1

        for slice in range(len(pred)):
            slice_pred = pred[slice]
            slice_GT = GT[slice]
            if slice_GT.max() != 0:
                
                for row in range(len(slice_pred)):
                    row_pred = slice_pred[row]
                    row_GT = slice_GT[row]
                    if row_GT.max() != 0:
                        try : 
                            auc.append(metrics.roc_auc_score(row_GT, row_pred))
                            f1.append(metrics.f1_score(row_GT, row_pred))
                            acc.append(metrics.accuracy_score(row_GT, row_pred))
                            sensitivity.append(metrics.recall_score(row_GT, row_pred))
                            precision.append(metrics.precision_score(row_GT, row_pred))
                        except:
                            # print('error slice:', slice, 'row:', row)
                            pass


    line = pd.DataFrame([[sum(val)/len(val) for val in [auc,f1,acc,sensitivity,precision,[cv_fold]]]], columns=model_params)
    Index_line = pd.MultiIndex.from_arrays([[idx_nbr+1.5],[model_name]], names=('number', 'name'))
    line.set_index(Index_line, inplace=True)
    Excel_Metrics = Excel_Metrics.append(line, ignore_index=False)
    Excel_Metrics = Excel_Metrics.sort_index()
    Excel_Metrics = Excel_Metrics.set_index(Excel_Metrics.index.droplevel('number').rename('Params'))

    # print(Excel_Metrics)

    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    Excel_Metrics.to_excel(writer, sheet_name=sheet, float_format="%.4f", header=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet]

    row_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
    for ind, row in enumerate(Excel_Metrics.index):
        if row in ['Params', 'Model\Metrics']:
            worksheet.set_row(ind, 15, row_format)

    col_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    for ind, col in enumerate(Excel_Metrics.columns):
        column_len = Excel_Metrics[col].astype(str).str.len().max()
        column_len = max(column_len, len(col)) + 2
        worksheet.set_column(ind+1, ind+1, column_len, col_format)

    indexcol_len = Excel_Metrics.index.astype(str).str.len().max()
    indexcol_len = max(indexcol_len, len(Excel_Metrics.index)) + 2
    worksheet.set_column(0, 0, indexcol_len, col_format)

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
    output_params.add_argument('--sheet_name', type=str, help='Name of the excel sheet to write on', default='Sheet1')

    training_parameters = parser.add_argument_group('Training parameters')
    training_parameters.add_argument('--model_name', type=str, help='name of the model', default='CBCT_seg_model')
    training_parameters.add_argument('--epochs', type=int, help='name of the model', default=20)
    training_parameters.add_argument('--batch_size', type=int, help='batch_size value', default=32)
    training_parameters.add_argument('--learning_rate', type=float, help='', default=0.0001)
    training_parameters.add_argument('--number_filters', type=int, help='', default=64)
    training_parameters.add_argument('--neighborhood', type=int, choices=[1,3,5,7,9], help='neighborhood slices (3|5|7)', default=3)
    training_parameters.add_argument('--cv_fold', type=int, help='number of the cross-validation fold', default=1)


    args = parser.parse_args()

    main(args)
