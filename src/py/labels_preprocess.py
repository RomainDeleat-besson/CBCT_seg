import argparse
import glob
import os

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
			if os.path.isfile(img_fn) and True in [ext in img_fn for ext in [".nrrd", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
				img_obj = {}
				img_obj["img"] = img_fn
				img_obj["out"] = os.path.normpath("/".join([args.out]))
				img_fn_array.append(img_obj)


	for img_obj in img_fn_array:
		image = img_obj["img"]
		out = img_obj["out"]

		if not os.path.exists(out):
			os.makedirs(out)
		
		img, _ = ReadFile(image, verbose=1)
		img = Normalize(np.array(img))
		
		print("Deconstruction...")
		Deconstruction(img, image, out, desired_width, desired_height)



if __name__ ==  '__main__':
	parser = argparse.ArgumentParser(description='Label pre-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_group = parser.add_argument_group('Input files')
	input_params = input_group.add_mutually_exclusive_group(required=True)
	input_params.add_argument('--image', type=str, help='Input 3D label')
	input_params.add_argument('--dir', type=str, help='Input directory with 3D labels')

	size_group = parser.add_argument_group('Resizing parameters')
	size_group.add_argument('--desired_width', type=int, default=512)
	size_group.add_argument('--desired_height', type=int, default=512)

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--out', type=str, help='Output directory of the label slices')

	args = parser.parse_args()

	main(args)
