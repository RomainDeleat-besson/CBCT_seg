import argparse
import os

import itk
import nibabel as nib
import numpy as np

from utils import *


def main(args):

	dir = args.dir
	original_dir = args.original_dir
	out = args.out

	if not os.path.exists(out):
		os.makedirs(out)

	filenames = []
	normpath = os.path.normpath("/".join([dir, '*', '']))
	for img_fn in glob.iglob(normpath, recursive=True):
		if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".png"]]:
			filename = os.path.normpath('_'.join(os.path.basename(img_fn).split('_')[:-1]))
			if filename not in filenames: filenames.append(filename)
	# print(filenames)

	original_img_paths = []
	for filename in filenames:
		normpath = os.path.normpath("/".join([original_dir, '**',filename+'*']))
		for img_fn in glob.iglob(normpath, recursive=True):
			if filename in img_fn: original_img_paths.append(img_fn)
	# print(original_img_paths)

	for (filename,original_img_path) in zip(filenames,original_img_paths):
		
		original_img, original_header = ReadFile(original_img_path)
		ext=''
		if '.nii' in original_img_path: ext='.nii'
		if '.gipl' in original_img_path: ext='.gipl'
		if '.nrrd' in original_img_path: ext='.nrrd'
		if '.gz' in original_img_path: ext=ext+'.gz'

		img = Reconstruction(filename,dir,original_img,out)
		# header = GetHeader(original_img, original_header, ext)

		outfile = os.path.normpath('/'.join([out,filename+'_rec'+ext]))
		SaveFile(outfile, img, original_header)
		

if __name__ ==  '__main__':
	parser = argparse.ArgumentParser(description='Post-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_params = parser.add_argument_group('Input files')
	input_params.add_argument('--dir', type=str, help='Input directory with 2D images', required=True)
	input_params.add_argument('--original_dir', type=str, help='Input directory with original 3D images', required=True)

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--out', type=str, help='Output directory', required=True)

	args = parser.parse_args()

	main(args)
