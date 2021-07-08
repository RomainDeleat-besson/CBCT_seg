import argparse
import glob
import os
import random
import shutil

from utils import *


def main(args):
    outdir = os.path.normpath("/".join([args.out]))
    train_path = os.path.join(outdir,'Training')
    test_path = os.path.join(outdir,'Testing')

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(test_path):
        os.makedirs(os.path.join(test_path,'Scans'))
        os.makedirs(os.path.join(test_path,'Segs'))
    if args.cv_folds == 0:
        os.makedirs(os.path.join(train_path,'Scans'))
        os.makedirs(os.path.join(train_path,'Segs'))
    else:
        for fold in range(args.cv_folds):
            folds_normpath = os.path.normpath("/".join([train_path,str(fold+1)]))
            if not os.path.exists(folds_normpath):
                os.makedirs(os.path.join(folds_normpath,'Scans'))
                os.makedirs(os.path.join(folds_normpath,'Segs'))

    fold = 0
    dirs_normpath = os.path.normpath("/".join([args.dir, '*', '']))
    nbr_files = 0
    for files in [args.dir+'/**/*'+ext for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
        nbr_files += len(glob.glob(files, recursive=True))
    nbr_scans = nbr_files/2
    
    for dir in glob.iglob(dirs_normpath, recursive=True):
        scan_fn_array = []
        seg_fn_array = []

        normpath = os.path.normpath("/".join([dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                img_obj = {}
                if True in [seg in os.path.basename(img_fn) for seg in ["seg","Seg"]]:
                    img_obj["img"] = img_fn
                    img_obj["out"] = os.path.join('Segs',os.path.basename(dir)+'_'+os.path.basename(img_fn))
                    seg_fn_array.append(img_obj)
                else:
                    img_obj["img"] = img_fn
                    img_obj["out"] = os.path.join('Scans',os.path.basename(dir)+'_'+os.path.basename(img_fn))
                    scan_fn_array.append(img_obj)
        
        if args.testing_number: test_nbr = round(args.testing_number*len(scan_fn_array)/nbr_scans)
        else: test_nbr = round(args.testing_percentage*len(scan_fn_array)/100)

        if args.validation_number: val_nbr = round(args.validation_number*len(scan_fn_array)/nbr_scans)
        else: val_nbr = round(args.validation_percentage*len(scan_fn_array)/100)

        for i in range(test_nbr):
            nbr = random.randint(0, len(scan_fn_array)-1)
            scan = scan_fn_array[nbr]["img"]
            seg = seg_fn_array[nbr]["img"]
            out_scan = os.path.join(test_path,scan_fn_array[nbr]["out"])
            out_seg = os.path.join(test_path,seg_fn_array[nbr]["out"])
            shutil.move(scan,out_scan)
            shutil.move(seg,out_seg)
            del scan_fn_array[nbr]
            del seg_fn_array[nbr]
        
        if args.cv_folds == 0:
            if val_nbr != 0:
                val_path = os.path.join(outdir,'Validation')
                if not os.path.exists(val_path):
                    os.makedirs(os.path.join(val_path,'Scans'))
                    os.makedirs(os.path.join(val_path,'Segs'))
                for i in range(val_nbr):
                    nbr = random.randint(0, len(scan_fn_array)-1)
                    scan = scan_fn_array[nbr]["img"]
                    seg = seg_fn_array[nbr]["img"]
                    out_scan = os.path.join(val_path,scan_fn_array[nbr]["out"])
                    out_seg = os.path.join(val_path,seg_fn_array[nbr]["out"])
                    shutil.move(scan,out_scan)
                    shutil.move(seg,out_seg)
                    del scan_fn_array[nbr]
                    del seg_fn_array[nbr]

            for (scan_obj,seg_obj) in zip(scan_fn_array,seg_fn_array):
                scan = scan_obj["img"]
                seg = seg_obj["img"]
                out_scan = os.path.join(train_path,scan_obj["out"])
                out_seg = os.path.join(train_path,seg_obj["out"])
                shutil.move(scan,out_scan)
                shutil.move(seg,out_seg)

        else:
            for (scan_obj,seg_obj) in zip(scan_fn_array,seg_fn_array):
                fold_path = os.path.normpath("/".join([train_path,str(fold+1)]))
                scan = scan_obj["img"]
                seg = seg_obj["img"]
                out_scan = os.path.join(fold_path,scan_obj["out"])
                out_seg = os.path.join(fold_path,seg_obj["out"])
                shutil.move(scan,out_scan)
                shutil.move(seg,out_seg)
                fold+=1
                if fold == args.cv_folds: fold=0

    if args.dir != args.out:
        shutil.rmtree(args.dir)



if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Creation of the workspace (training, validation, testing | cross validation)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_params = parser.add_argument_group('Input files')
    input_params.add_argument('--dir', type=str, help='Input directory with 3D images', required=True)

    output_params = parser.add_argument_group('Output parameters')
    output_params.add_argument('--out', type=str, help='Output directory', required=True)
    output_params.add_argument('--cv_folds', type=int, help='Number of folds to create', default=8)
    
    testing_params = output_params.add_mutually_exclusive_group()
    testing_params.add_argument('--testing_number', type=int, help='Number of scans to keep for testing')
    testing_params.add_argument('--testing_percentage', type=int, help='Percentage of scans to keep for testing', default=20)

    validation_params = output_params.add_mutually_exclusive_group()
    validation_params.add_argument('--validation_number', type=int, help='Number of scans to keep for validation')
    validation_params.add_argument('--validation_percentage', type=int, help='Percentage of scans to keep for validation', default=10)

    args = parser.parse_args()

    main(args)
