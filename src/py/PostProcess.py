import argparse
import os

import itk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.filters import threshold_local, threshold_otsu

from sklearn.cluster import KMeans
from collections import Counter
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
		thresh = threshold_otsu(img)
		img = img > thresh
		img = img.astype(np.ushort)

		ImageType = itk.Image[itk.US, 3]
		itk_img = itk.PyBuffer[ImageType].GetImageFromArray(img)



		ConnectedComponentImageFilter = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
		ConnectedComponentImageFilter.SetInput(itk_img)
		ConnectedComponentImageFilter.Update()

		RelabelComponentImageFilter = itk.RelabelComponentImageFilter[ImageType, ImageType].New()
		RelabelComponentImageFilter.SetInput(ConnectedComponentImageFilter)
		RelabelComponentImageFilter.SetMinimumObjectSize(800)
		RelabelComponentImageFilter.Update()

		print("Max pixel object:", RelabelComponentImageFilter.GetSizeOfObjectsInPixels()[0])
		print("Min pixel object:", RelabelComponentImageFilter.GetSizeOfObjectsInPixels()[-1])
		print("Number of object before:", RelabelComponentImageFilter.GetOriginalNumberOfObjects())
		print("Number of object after :", RelabelComponentImageFilter.GetNumberOfObjects())

		labelStatisticsImageFilter = itk.LabelStatisticsImageFilter[ImageType, ImageType].New()
		labelStatisticsImageFilter.SetLabelInput(RelabelComponentImageFilter.GetOutput())
		labelStatisticsImageFilter.SetInput(itk_img)
		labelStatisticsImageFilter.Update()
		NumberOfLabel = len(labelStatisticsImageFilter.GetValidLabelValues())
		

		L_BoundingBoxSlices = []
		for i in range (1, NumberOfLabel):
			xmin = labelStatisticsImageFilter.GetBoundingBox(i)[0]
			xmax = labelStatisticsImageFilter.GetBoundingBox(i)[1]
			L_BoundingBoxSlices.append([xmin, xmax])

		L_BoundingBoxSlices = np.array(L_BoundingBoxSlices)
		kmeans = KMeans(n_clusters=2).fit(L_BoundingBoxSlices)

		NumberCluster = Counter(kmeans.labels_)
		mean_cluster_0 = np.mean(kmeans.cluster_centers_[0])
		mean_cluster_1 = np.mean(kmeans.cluster_centers_[1])

		if mean_cluster_0 > mean_cluster_1:
			indice_upper=0
			indice_lower=1
		else:
			indice_upper=1
			indice_lower=0


		LabelType = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,3]]
		LabelImageToLabelMapFilter = itk.LabelImageToLabelMapFilter[ImageType, LabelType].New()
		LabelImageToLabelMapFilter.SetInput(RelabelComponentImageFilter)
		LabelImageToLabelMapFilter.Update()

		print("c1   ", NumberCluster[0])
		print("c2   ", NumberCluster[1])
		print("c1/c2", NumberCluster[0]/NumberCluster[1])

		# Condition to know if there are 2 jaws in the scan
		if NumberOfLabel>20 and (NumberCluster[0]/NumberCluster[1] > 0.6 and NumberCluster[0]/NumberCluster[1] < 1.4):
			RelabelComponentImageFilter_upper = itk.image_duplicator(RelabelComponentImageFilter)
			RelabelComponentImageFilter_lower = itk.image_duplicator(RelabelComponentImageFilter)

			LabelImageToLabelMapFilter_upper = itk.LabelImageToLabelMapFilter[ImageType, LabelType].New()
			LabelImageToLabelMapFilter_upper.SetInput(RelabelComponentImageFilter_upper)
			LabelImageToLabelMapFilter_upper.Update()

			LabelImageToLabelMapFilter_lower = itk.LabelImageToLabelMapFilter[ImageType, LabelType].New()
			LabelImageToLabelMapFilter_lower.SetInput(RelabelComponentImageFilter_lower)
			LabelImageToLabelMapFilter_lower.Update()

			LabelImageToLabelMapFilter_upper = LabelImageToLabelMapFilter_upper.GetOutput()
			LabelImageToLabelMapFilter_lower = LabelImageToLabelMapFilter_lower.GetOutput()


			for i in range (1, NumberOfLabel):
				if kmeans.labels_[i-1] == indice_upper:
					LabelImageToLabelMapFilter_lower.RemoveLabel(i)

				if kmeans.labels_[i-1] == indice_lower:
					LabelImageToLabelMapFilter_upper.RemoveLabel(i)


			LabelMapToLabelImageFilter_upper = itk.LabelMapToLabelImageFilter[LabelType, ImageType].New()
			LabelMapToLabelImageFilter_upper.SetInput(LabelImageToLabelMapFilter_upper)
			LabelMapToLabelImageFilter_upper.Update()

			LabelMapToLabelImageFilter_lower = itk.LabelMapToLabelImageFilter[LabelType, ImageType].New()
			LabelMapToLabelImageFilter_lower.SetInput(LabelImageToLabelMapFilter_lower)
			LabelMapToLabelImageFilter_lower.Update()

			outfile = os.path.normpath('/'.join([out,filename+'_'+"upper_"+tool_name+ext]))
			SaveFile(outfile, LabelMapToLabelImageFilter_upper.GetOutput(), original_header)
			
			outfile = os.path.normpath('/'.join([out,filename+'_'+"lower_"+tool_name+ext]))
			SaveFile(outfile, LabelMapToLabelImageFilter_lower.GetOutput(), original_header)
		
		else:
			LabelMapToLabelImageFilter = itk.LabelMapToLabelImageFilter[LabelType, ImageType].New()
			LabelMapToLabelImageFilter.SetInput(LabelImageToLabelMapFilter)
			LabelMapToLabelImageFilter.Update()

			outfile = os.path.normpath('/'.join([out,filename+'_'+tool_name+ext]))
			SaveFile(outfile, LabelMapToLabelImageFilter.GetOutput(), original_header)

		outfile = os.path.normpath('/'.join([out,filename+'_raw_'+tool_name+ext]))
		SaveFile(outfile, ConnectedComponentImageFilter.GetOutput(), original_header)		



if __name__ ==  '__main__':
	parser = argparse.ArgumentParser(description='Post-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_params = parser.add_argument_group('Input files')
	input_params.add_argument('--dir', type=str, help='Input directory with 2D images', required=True)
	input_params.add_argument('--original_dir', type=str, help='Input directory with original 3D images', required=True)

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--tool', type=str, help='Name of the tool used', default='RCSeg')
	output_params.add_argument('--out', type=str, help='Output directory', required=True)

	args = parser.parse_args()

	main(args)
