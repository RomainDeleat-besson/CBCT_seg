import argparse
import os

import itk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.filters import threshold_local, threshold_otsu

from utils import *


def main(args):

	dir = args.dir
	original_dir = args.original_dir
	tool_name = args.tool
	out = args.out

	if not os.path.exists(out):
		os.makedirs(out)

	original_img_paths = [ori_fn for ori_fn in glob.iglob(os.path.normpath("/".join([original_dir, '*', ''])), recursive=True)]

	for original_img_path in glob.iglob(os.path.normpath("/".join([original_dir, '*', ''])), recursive=True):
		
		basename = os.path.basename(original_img_path).split('.')[0]
		filename = '_'.join([file for file in os.listdir(dir) if file.startswith(basename)][0].split('_')[:-1])

		original_img, original_header = ReadFile(original_img_path)
		ext=''
		if '.nii' in original_img_path: ext='.nii'
		if '.gipl' in original_img_path: ext='.gipl'
		if '.nrrd' in original_img_path: ext='.nrrd'
		if '.gz' in original_img_path: ext=ext+'.gz'

		img = Reconstruction(filename,dir,original_img,out)
		img = img.astype(np.ushort)

		ImageType = itk.Image[itk.US, 3]
		itk_img = itk.PyBuffer[ImageType].GetImageFromArray(img)

		OtsuThresholdImageFilter = itk.OtsuThresholdImageFilter[ImageType, ImageType].New()
		OtsuThresholdImageFilter.SetInput(itk_img)
		OtsuThresholdImageFilter.SetInsideValue(0)
		OtsuThresholdImageFilter.SetOutsideValue(1)
		OtsuThresholdImageFilter.Update()
		binary_img = OtsuThresholdImageFilter.GetOutput()

		label = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
		label.SetInput(binary_img)
		label.Update()

		labelStatisticsImageFilter = itk.LabelStatisticsImageFilter[ImageType, ImageType].New()
		labelStatisticsImageFilter.SetLabelInput(label.GetOutput())
		labelStatisticsImageFilter.SetInput(binary_img)
		labelStatisticsImageFilter.Update()
		labelList = labelStatisticsImageFilter.GetValidLabelValues()
		NbreOfLabel = len(labelList)

		if tool_name == 'MandSeg':
			labelSize=[]
			[labelSize.append(labelStatisticsImageFilter.GetCount(i)) for i in range(1,len(labelList))]
			Max = max(labelSize)

			RelabelComponentImageFilter = itk.RelabelComponentImageFilter[ImageType, ImageType].New()
			RelabelComponentImageFilter.SetInput(label)
			RelabelComponentImageFilter.SetMinimumObjectSize(Max)
			RelabelComponentImageFilter.Update()
			relabeled_itk_img = RelabelComponentImageFilter.GetOutput()

		# outfile = os.path.normpath('/'.join([out,filename+'_Original_'+tool_name+ext]))
		# SaveFile(outfile, itk.GetArrayFromImage(itk_img), original_header)
		outfile = os.path.normpath('/'.join([out,filename+'_'+tool_name+ext]))
		SaveFile(outfile, itk.GetArrayFromImage(binary_img), original_header)
		

if __name__ ==  '__main__':
	parser = argparse.ArgumentParser(description='Post-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_params = parser.add_argument_group('Input files')
	input_params.add_argument('--dir', type=str, help='Input directory with 2D images', required=True)
	input_params.add_argument('--original_dir', type=str, help='Input directory with original 3D images', required=True)

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--tool', type=str, help='Name of the tool used', default='MandSeg')
	output_params.add_argument('--out', type=str, help='Output directory', required=True)

	args = parser.parse_args()

	main(args)
