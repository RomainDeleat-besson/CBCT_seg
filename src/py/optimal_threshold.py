import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
from sklearn.metrics import log_loss, precision_score, recall_score

from utils import *


@jit(nopython=True, nogil=True, cache=True, parallel=True, fastmath=True)
def binary_cross_entropy (y_true, y_pred):
    out = 0
    for i in prange(y_pred.size):
        out += y_true[i] * np.log(y_pred[i]+1e-15) + (1 - y_true[i])*np.log(1-y_pred[i]+1e-15)
    return - out / y_pred.size

@jit(nopython=True, nogil=True, cache=True, parallel=True, fastmath=True)
def compute_tp_fn_fp (y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for i in prange(y_pred.size):
        tp += y_true[i] * y_pred[i]
        fp += (1-y_true[i]) * y_pred[i]
        fn += y_true[i] * (1-y_pred[i])
        
    return tp, fp, fn
    
def compute_precision(tp, fp):
    return tp / (tp + fp)

def compute_recall(tp, fn):
    return tp / (tp + fn)

def compute_f1_score(precision, recall):
    try:
        return (2*precision*recall) / (precision + recall)
    except:
        return 0

def fastness():
    np.random.seed(123)
    y_pred = np.uint8(np.random.uniform(low=0.0, high=1.0, size=1_000_000) > 0.5)
    y_true = np.uint8(np.random.randint(low=0, high=2, size=1_000_000))

    y_pred[:200000] = 1
    y_true[:200000] = 1

    print(y_pred[:10])
    print(y_true[:10])
    print(np.log(y_pred)[:10])

    print("=== Loss ===")
    T0 = time.time()
    loss_sklearn = log_loss(y_true, y_pred)
    T1 = time.time()
    loss_numpy = binary_cross_entropy(y_true, y_pred)
    T2 = time.time()

    print(loss_sklearn, round(T1-T0, 5))
    print(loss_numpy, round(T2-T1, 5))
    print(f"Numpy is {(T1-T0)/(T2-T1):.1f}x faster than sklearn\n")

    print("=== Precision & Recall ===")

    T0 = time.time()
    precision_sklearn = precision_score(y_true, y_pred, pos_label=1)
    recall_sklearn = recall_score(y_true, y_pred, pos_label=1)
    T1 = time.time()
    tp, fp, fn = compute_tp_fn_fp(y_true, y_pred)
    precision_numpy = compute_precision(tp, fp)
    recall_numpy = compute_recall(tp, fn)
    T2 = time.time()

    print(precision_sklearn, recall_sklearn, round(T1-T0, 5)) # Inversed ???
    print(precision_numpy, recall_numpy, round(T2-T1, 5))
    print(f"Numpy is {(T1-T0)/(T2-T1):.1f}x faster than sklearn\n")
    print()


def main(args):
    #====== Numba compilation ======
    # The 2 lines are important for both cases
    binary_cross_entropy(np.array([0,0,0], dtype=np.uint8), np.array([0,1,0], dtype=np.uint8))
    binary_cross_entropy(np.array([0,0,0], dtype=np.float32), np.array([0,1,0], dtype=np.float32))

    compute_tp_fn_fp(np.array([0,0,0], dtype=np.uint8), np.array([0,1,0], dtype=np.uint8))
    compute_tp_fn_fp(np.array([0,0,0], dtype=np.float32), np.array([0,1,0], dtype=np.float32))
    #===============================

    dir = args.dir
    original_dir = args.original_dir
    groundtruth_dir = args.groundtruth

    L_threshold_mean = []
    L_loss_mean = []
    L_F1score_mean =[]
    L_recall_mean = []
    L_precision_mean = []

    for (original_img_path, groundtruth_img_path) in zip(sorted(glob.iglob(os.path.normpath("/".join([original_dir, '*', ''])), recursive=True)), sorted(glob.iglob(os.path.normpath("/".join([groundtruth_dir, '*', ''])), recursive=True))):
        print("============================================")
        start_time = time.time()

        basename = os.path.basename(original_img_path).split('.')[0]
        filename = '_'.join([file for file in os.listdir(dir) if file.startswith(basename)][0].split('_')[:-1])

        original_img, original_header = ReadFile(original_img_path)
        ext=''
        if '.nii' in original_img_path: ext='.nii'
        if '.gipl' in original_img_path: ext='.gipl'
        if '.nrrd' in original_img_path: ext='.nrrd'
        if '.gz' in original_img_path: ext=ext+'.gz'

        groundtruth, _ = ReadFile(groundtruth_img_path)
        img = Reconstruction(filename,dir,original_img)

        if args.verbose_fastness:
            fastness()

        L_threshold = []
        L_loss = []
        L_F1score = []
        L_recall = []
        L_precision = []  

        pred = np.array(img).flatten()
        labels = np.array(groundtruth).flatten()
        labels = np.uint8(labels > 0.5)

        for thresh in range (1, 251):
            pred_thresh = np.uint8(pred > thresh)

            tp, fp, fn = compute_tp_fn_fp(labels, pred_thresh)
            recall = compute_recall(tp, fn)
            precision = compute_precision(tp, fp)
            f1 = compute_f1_score(precision, recall)
            loss = binary_cross_entropy(labels, pred_thresh)

            L_threshold.append(thresh)
            L_recall.append(recall)
            L_precision.append(precision)
            L_F1score.append(f1)
            L_loss.append(loss)

        L_threshold_mean.append(L_threshold)
        L_recall_mean.append(L_recall)
        L_precision_mean.append(L_precision) 
        L_F1score_mean.append(L_F1score) 
        L_loss_mean.append(L_loss)
        end_time = time.time()
        print("Time for one scan: {0:.2f} seconds".format(end_time - start_time))


    L_threshold_mean = np.mean(np.array(L_threshold_mean), axis=0)
    L_recall_mean    = np.mean(np.array(L_recall_mean),    axis=0)
    L_precision_mean = np.mean(np.array(L_precision_mean), axis=0)
    L_F1score_mean   = np.mean(np.array(L_F1score_mean),   axis=0)
    L_loss_mean      = np.mean(np.array(L_loss_mean),      axis=0)


    # with open('saved_arr.npy', 'wb') as file:
    #     np.save(file, L_threshold_mean)
    #     np.save(file, L_recall_mean)
    #     np.save(file, L_precision_mean)
    #     np.save(file, L_F1score_mean)
    #     np.save(file, L_loss_mean)

    # with open('saved_arr.npy', 'rb') as file:
    #     L_threshold_mean = np.load(file)
    #     L_recall_mean = np.load(file)
    #     L_precision_mean = np.load(file)
    #     L_F1score_mean = np.load(file)
    #     L_loss_mean = np.load(file)


    L_recall_down    = L_recall_mean    - np.std(np.array(L_recall_mean),    axis=0)
    L_precision_down = L_precision_mean - np.std(np.array(L_precision_mean), axis=0)
    L_F1score_down   = L_F1score_mean   - np.std(np.array(L_F1score_mean),   axis=0)
    L_loss_down      = L_loss_mean      - np.std(np.array(L_loss_mean),      axis=0)

    L_recall_up      = L_recall_mean    + np.std(np.array(L_recall_mean),    axis=0)
    L_precision_up   = L_precision_mean + np.std(np.array(L_precision_mean), axis=0)
    L_F1score_up     = L_F1score_mean   + np.std(np.array(L_F1score_mean),   axis=0)
    L_loss_up        = L_loss_mean      + np.std(np.array(L_loss_mean),      axis=0)

    # L_recall_down    = np.amin(np.array(L_recall_mean),    axis=0)
    # L_precision_down = np.amin(np.array(L_precision_mean), axis=0)
    # L_F1score_down   = np.amin(np.array(L_F1score_mean),   axis=0)
    # L_loss_down      = np.amin(np.array(L_loss_mean),      axis=0)

    # L_recall_up      = np.amax(np.array(L_recall_mean),    axis=0)
    # L_precision_up   = np.amax(np.array(L_precision_mean), axis=0)
    # L_F1score_up     = np.amax(np.array(L_F1score_mean),   axis=0)
    # L_loss_up        = np.amax(np.array(L_loss_mean),      axis=0)


    plt.figure(figsize=(13,8))

    plt.subplot(221)
    plt.fill_between(L_threshold_mean, L_loss_down, L_loss_up, color='k', alpha=0.2)
    plt.plot(L_threshold_mean, L_loss_mean, c='k', lw=1)
    plt.hlines(0, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.ylim([-0.005, 0.11])
    plt.grid(True)
    plt.title("Binary-CrossEntropy Loss")
    plt.xlabel("Loss")
    plt.ylabel("Threshold")

    plt.subplot(222)
    plt.fill_between(L_threshold_mean, L_F1score_down, L_F1score_up, color='r', alpha=0.2)
    plt.plot(L_threshold_mean, L_F1score_mean, c='r', lw=1)
    plt.hlines(0, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.hlines(1, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.ylim([0, 1.1])
    plt.grid(True)
    plt.title("F1 score")
    plt.xlabel("F1 score")
    plt.ylabel("Threshold")

    plt.subplot(223)
    plt.fill_between(L_threshold_mean, L_recall_down, L_recall_up, color='g', alpha=0.2)
    plt.plot(L_threshold_mean, L_recall_mean, c='g', lw=1)
    plt.hlines(0, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.hlines(1, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.ylim([0, 1.1])
    plt.grid(True)
    plt.title("Recall")
    plt.xlabel("Recall")
    plt.ylabel("Threshold")

    plt.subplot(224)
    plt.fill_between(L_threshold_mean, L_precision_down, L_precision_up, color='b', alpha=0.2)
    plt.plot(L_threshold_mean, L_precision_mean, c='b', lw=1)
    plt.hlines(0, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.hlines(1, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.ylim([0, 1.1])
    plt.grid(True)
    plt.title("Precision")
    plt.xlabel("Precision")
    plt.ylabel("Threshold")

    plt.plot()


    ##################################################################
    # Regroupé en 2 plot
    ##################################################################

    plt.figure(figsize=(13,8))

    plt.subplot(121)
    plt.fill_between(L_threshold_mean, L_loss_down, L_loss_up, color='k', alpha=0.2)
    plt.plot(L_threshold_mean, L_loss_mean, c='k', lw=1)
    plt.hlines(0, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.ylim([-0.005, 0.11])
    plt.grid(True)
    plt.title("Binary-CrossEntropy Loss")
    plt.xlabel("Loss")
    plt.ylabel("Threshold")

    plt.subplot(122)
    plt.fill_between(L_threshold_mean, L_F1score_down, L_F1score_up, color='r', alpha=0.2)
    plt.fill_between(L_threshold_mean, L_recall_down, L_recall_up, color='g', alpha=0.2)
    plt.fill_between(L_threshold_mean, L_precision_down, L_precision_up, color='b', alpha=0.2)
    plt.plot(L_threshold_mean, L_F1score_mean, c='r', lw=1)
    plt.plot(L_threshold_mean, L_recall_mean, c='g', lw=1)
    plt.plot(L_threshold_mean, L_precision_mean, c='b', lw=1)
    plt.hlines(0, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.hlines(1, L_threshold_mean[0], L_threshold_mean[-1], colors='gray')
    plt.ylim([0, 1.1])
    plt.grid(True)
    plt.title("F1 score, Recall, Precision")
    plt.ylabel("Threshold")
    plt.legend(["F1 score", "Recall", "Precision"])

    plt.show()


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Optimal threshold', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_params = parser.add_argument_group('Input files')
    input_params.add_argument('--dir', type=str, help='Input directory with 2D images', required=True)
    input_params.add_argument('--original_dir', type=str, help='Input directory with original 3D scans', required=True)
    input_params.add_argument('--groundtruth', type=str, help='Input directory with original 3D labels', required=True)

    input_params = parser.add_argument_group('Input files')
    input_params.add_argument('--verbose_fastness', type=bool, help='print how fast is the method', default=False)


    args = parser.parse_args()

    main(args)





