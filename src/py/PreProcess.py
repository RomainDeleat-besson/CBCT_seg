import argparse
import glob
import os

import itk
import numpy as np

from utils import *


def main(args):

	desired_width = args.desired_width
	desired_height = args.desired_height

	img_fn_array = []

	if args.image:
		img_obj = {}
		img_obj["img"] = args.image
		img_obj["out"] = args.out
		img_fn_array.append(img_obj)

	if args.dir:
		normpath = os.path.normpath("/".join([args.dir, '*', '']))
		for img_fn in glob.iglob(normpath, recursive=True):
			if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
				img_obj = {}
				img_obj["img"] = img_fn
				img_obj["out"] = os.path.normpath("/".join([args.out]))
				img_fn_array.append(img_obj)


	for img_obj in img_fn_array:
		image = img_obj["img"]
		out = img_obj["out"]

		if not os.path.exists(out):
			os.makedirs(out)
		# else:
		# 	shutil.rmtree(out)
		# 	os.makedirs(out)
		
		img, header = ReadFile(image)
		
		print("Normalization and contrast adjustment...")

		img = Normalize(img,in_min=0,in_max=img.max(),out_min=0,out_max=255)
		img = Adjust_Contrast(img,pmin=45,pmax=90)

		Deconstruction(img, image, out, desired_width, desired_height)

		





if __name__ ==  '__main__':
	parser = argparse.ArgumentParser(description='Pre-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_group = parser.add_argument_group('Input file')
	input_params = input_group.add_mutually_exclusive_group(required=True)
	input_params.add_argument('--image', type=str, help='Input 3D image')
	input_params.add_argument('--dir', type=str, help='Input directory with 3D images')

	size_group = parser.add_argument_group('Resizing parameters')
	size_group.add_argument('--desired_width', type=int, default=512)
	size_group.add_argument('--desired_height', type=int, default=512)

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--out', type=str, help='Output directory')

	args = parser.parse_args()

	main(args)
