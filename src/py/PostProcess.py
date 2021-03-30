import argparse
import os

import itk
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

	ImageType = itk.Image[itk.F, 3]
	for (filename,original_img_path) in zip(filenames,original_img_paths):
		
		print('Reconstruction: ',filename)
		original_img = ReadFile(original_img_path, verbose=0)
		img = Reconstruction(filename,dir,original_img,out)
		img = img.astype(np.float32)
		img = itk.PyBuffer[ImageType].GetImageFromArray(img)

		img = Resample(img, original_img_path)
		outfile = os.path.normpath('/'.join([out,filename+'.nrrd']))
		SaveFile(outfile, img, ImageType)



if __name__ ==  '__main__':
	parser = argparse.ArgumentParser(description='Post-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_params = parser.add_argument_group('Input file')
	input_params.add_argument('--dir', type=str, help='Input directory with 2D images', required=True)
	input_params.add_argument('--original_dir', type=str, help='Input directory with original 3D images', required=True)

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--out', type=str, help='Output directory', required=True)

	args = parser.parse_args()

	main(args)
