import argparse
import os
from collections import Counter

import itk
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans

from utils import *


def main(args):
	dir = args.dir
	original_dir = args.original_dir
	tool_name = args.tool
	out = args.out

	if not os.path.exists(out):
		os.makedirs(out)

	for original_img_path in glob.iglob(os.path.normpath("/".join([original_dir, '*', ''])), recursive=True):
		print("============================================")
		
		basename = os.path.basename(original_img_path).split('.')[0]
		filename = '_'.join([file for file in os.listdir(dir) if file.startswith(basename)][0].split('_')[:-1])

		original_img, original_header = ReadFile(original_img_path)
		ext=''
		if '.nii' in original_img_path: ext='.nii'
		if '.gipl' in original_img_path: ext='.gipl'
		if '.nrrd' in original_img_path: ext='.nrrd'
		if '.gz' in original_img_path: ext=ext+'.gz'

		raw_img = Reconstruction(filename,dir,original_img)
		img = raw_img.copy()

		if args.threshold == -1:
			thresh=int(round(threshold_otsu(img)))
		else:
			thresh=args.threshold
		
		print("Threshold:", thresh)
		img[img<thresh]=0
		img[img>=thresh]=255
		img = img.astype(np.ushort)

		ImageType = itk.Image[itk.US, 3]
		itk_img = itk.PyBuffer[ImageType].GetImageFromArray(img)

		ConnectedComponentImageFilter = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
		ConnectedComponentImageFilter.SetInput(itk_img)
		ConnectedComponentImageFilter.Update()

		if tool_name == 'RCSeg':
			try:
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

				distance_clusters = abs(mean_cluster_0 - mean_cluster_1)
				print("dist mean cluster:", distance_clusters)	

				# Condition to know if there are 2 jaws in the scan
				if distance_clusters > 40 and not (NumberCluster[0]<=6 or NumberCluster[1]<=6):
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

					LabelUpper, LabelLower = [], []
					for i in range (1, NumberOfLabel):
						if kmeans.labels_[i-1] == indice_upper:
							LabelImageToLabelMapFilter_lower.RemoveLabel(i)
							LabelUpper.append(i)

						if kmeans.labels_[i-1] == indice_lower:
							LabelImageToLabelMapFilter_upper.RemoveLabel(i)
							LabelLower.append(i)

					# Upper
					ChangeLabelLabelMapFilter_upper = itk.ChangeLabelLabelMapFilter[LabelType].New()
					ChangeLabelLabelMapFilter_upper.SetInput(LabelImageToLabelMapFilter_upper)

					for lbl in LabelUpper:
						ChangeLabelLabelMapFilter_upper.SetChange(lbl, 1)
					ChangeLabelLabelMapFilter_upper.Update()

					LabelMapToLabelImageFilter_upper = itk.LabelMapToLabelImageFilter[LabelType, ImageType].New()
					LabelMapToLabelImageFilter_upper.SetInput(ChangeLabelLabelMapFilter_upper)
					LabelMapToLabelImageFilter_upper.Update()

					outfile = os.path.normpath('/'.join([out,filename+'_'+"upper_"+tool_name+ext]))
					SaveFile(outfile, LabelMapToLabelImageFilter_upper.GetOutput(), original_header)
					
					# Lower
					ChangeLabelLabelMapFilter_lower = itk.ChangeLabelLabelMapFilter[LabelType].New()
					ChangeLabelLabelMapFilter_lower.SetInput(LabelImageToLabelMapFilter_lower)

					for lbl in LabelLower:
						ChangeLabelLabelMapFilter_lower.SetChange(lbl, 1)
					ChangeLabelLabelMapFilter_lower.Update()

					LabelMapToLabelImageFilter_lower = itk.LabelMapToLabelImageFilter[LabelType, ImageType].New()
					LabelMapToLabelImageFilter_lower.SetInput(ChangeLabelLabelMapFilter_lower)
					LabelMapToLabelImageFilter_lower.Update()
					
					outfile = os.path.normpath('/'.join([out,filename+'_'+"lower_"+tool_name+ext]))
					SaveFile(outfile, LabelMapToLabelImageFilter_lower.GetOutput(), original_header)
				
				else:
					if NumberCluster[0] > NumberCluster[1]:
						artifact = 1
					else:
						artifact = 0

					if distance_clusters > 40 :
						for i in range(NumberOfLabel-1):
							if kmeans.labels_[i] == artifact:
								LabelImageToLabelMapFilter.GetOutput().RemoveLabel(i+1)
					LabelImageToLabelMapFilter.Update()

					ChangeLabelLabelMapFilter = itk.ChangeLabelLabelMapFilter[LabelType].New()
					ChangeLabelLabelMapFilter.SetInput(LabelImageToLabelMapFilter)

					for lbl in range(1, NumberOfLabel):
						ChangeLabelLabelMapFilter.SetChange(lbl, 1)
					ChangeLabelLabelMapFilter.Update()
					
					LabelMapToLabelImageFilter = itk.LabelMapToLabelImageFilter[LabelType, ImageType].New()
					LabelMapToLabelImageFilter.SetInput(ChangeLabelLabelMapFilter)
					LabelMapToLabelImageFilter.Update()

					outfile = os.path.normpath('/'.join([out,filename+'_'+tool_name+ext]))
					SaveFile(outfile, LabelMapToLabelImageFilter.GetOutput(), original_header)

				if args.out_raw:
					if distance_clusters > 40 and not (NumberCluster[0]<=6 or NumberCluster[1]<=6):
						mean_jaws = int(abs((mean_cluster_0 + mean_cluster_1)/2))
						if "lower" in filename:
							raw_img[:,:,mean_jaws:] = 0

						elif 'upper' in filename:
							raw_img[:,:,:mean_jaws] = 0

					if not os.path.exists(args.out_raw):
						os.makedirs(args.out_raw)
					outfile = os.path.normpath('/'.join([args.out_raw,filename+'_raw_'+tool_name+ext]))
					SaveFile(outfile, raw_img, original_header)
				
			except:
				print("can't apply initial postprocess, saving raw prediction:")
				outfile = os.path.normpath('/'.join([out,filename+'_raw_'+tool_name+ext]))
				SaveFile(outfile, ConnectedComponentImageFilter.GetOutput(), original_header)


		else: #MandSeg
			labelStatisticsImageFilter = itk.LabelStatisticsImageFilter[ImageType, ImageType].New()
			labelStatisticsImageFilter.SetLabelInput(ConnectedComponentImageFilter.GetOutput())
			labelStatisticsImageFilter.SetInput(itk_img)
			labelStatisticsImageFilter.Update()
			labelList = labelStatisticsImageFilter.GetValidLabelValues()
			NumberOfLabel = len(labelList)

			labelSize=[]
			[labelSize.append(labelStatisticsImageFilter.GetCount(i)) for i in range(1,len(labelList))]
			# print(labelSize)
			Max = max(labelSize)
			# print("Max: ", Max)
			# print("Thresh: ", Max/20)

			RelabelComponentImageFilter = itk.RelabelComponentImageFilter[ImageType, ImageType].New()
			RelabelComponentImageFilter.SetInput(ConnectedComponentImageFilter)
			RelabelComponentImageFilter.SetMinimumObjectSize(int(Max/20))
			RelabelComponentImageFilter.Update()
			relabeled_itk_img = RelabelComponentImageFilter.GetOutput()

			LabelType = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,3]]
			LabelImageToLabelMapFilter = itk.LabelImageToLabelMapFilter[ImageType, LabelType].New()
			LabelImageToLabelMapFilter.SetInput(relabeled_itk_img)
			LabelImageToLabelMapFilter.Update()

			ChangeLabelLabelMapFilter = itk.ChangeLabelLabelMapFilter[LabelType].New()
			ChangeLabelLabelMapFilter.SetInput(LabelImageToLabelMapFilter)
			# NumberOfLabel = ChangeLabelLabelMapFilter.GetNumberOfIndexedOutputs()

			ConnectedComponentImageFilter = itk.ConnectedComponentImageFilter[ImageType, ImageType].New()
			ConnectedComponentImageFilter.SetInput(relabeled_itk_img)
			ConnectedComponentImageFilter.Update()

			labelStatisticsImageFilter = itk.LabelStatisticsImageFilter[ImageType, ImageType].New()
			labelStatisticsImageFilter.SetLabelInput(ConnectedComponentImageFilter.GetOutput())
			labelStatisticsImageFilter.SetInput(relabeled_itk_img)
			labelStatisticsImageFilter.Update()
			labelList = labelStatisticsImageFilter.GetValidLabelValues()
			NumberOfLabel = len(labelList)

			print("NumberOfLabel: ", NumberOfLabel)

			for lbl in range(1, NumberOfLabel):
				ChangeLabelLabelMapFilter.SetChange(lbl, 1)
			ChangeLabelLabelMapFilter.Update()
			
			LabelMapToLabelImageFilter = itk.LabelMapToLabelImageFilter[LabelType, ImageType].New()
			LabelMapToLabelImageFilter.SetInput(ChangeLabelLabelMapFilter)
			LabelMapToLabelImageFilter.Update()
			relabeled_itk_img = LabelMapToLabelImageFilter.GetOutput()


			StructuringElementType = itk.FlatStructuringElement[3]
			structuringElement = StructuringElementType.Ball(5)

			BinaryMorphologicalClosingImageFilter = itk.BinaryMorphologicalClosingImageFilter[ImageType,ImageType,StructuringElementType].New()
			BinaryMorphologicalClosingImageFilter.SetInput(relabeled_itk_img)
			BinaryMorphologicalClosingImageFilter.SetKernel(structuringElement)
			BinaryMorphologicalClosingImageFilter.SetForegroundValue(1)
			BinaryMorphologicalClosingImageFilter.Update()
			closed_itk_img = BinaryMorphologicalClosingImageFilter.GetOutput()

			BinaryFillholeImageFilter = itk.BinaryFillholeImageFilter[ImageType].New()
			BinaryFillholeImageFilter.SetInput(closed_itk_img)
			BinaryFillholeImageFilter.SetForegroundValue(1)
			BinaryFillholeImageFilter.Update()
			filled_itk_img = BinaryFillholeImageFilter.GetOutput()

			outfile = os.path.normpath('/'.join([out,filename+'_'+tool_name+ext]))
			SaveFile(outfile, itk.GetArrayFromImage(filled_itk_img), original_header)

			if args.out_raw:
				if not os.path.exists(args.out_raw):
					os.makedirs(args.out_raw)
				outfile = os.path.normpath('/'.join([args.out_raw,filename+'_raw_'+tool_name+ext]))
				SaveFile(outfile, raw_img, original_header)
	


if __name__ ==  '__main__':
	parser = argparse.ArgumentParser(description='Post-processing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	input_params = parser.add_argument_group('Input files')
	input_params.add_argument('--dir', type=str, help='Input directory with 2D images', required=True)
	input_params.add_argument('--original_dir', type=str, help='Input directory with original 3D images', required=True)

	param_parser = parser.add_argument_group('Parameters')
	param_parser.add_argument('--tool', type=str, help='Name of the tool used', default='MandSeg')
	param_parser.add_argument('--threshold', type=int, help='if -1, the thresold apply is otsu, else it is the value entered (between [0;255])', default=-1)

	output_params = parser.add_argument_group('Output parameters')
	output_params.add_argument('--out', type=str, help='Output directory', required=True)
	output_params.add_argument('--out_raw', type=str, help='Output directory for raw files')

	args = parser.parse_args()

	main(args)
