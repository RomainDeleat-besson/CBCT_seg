import os
import re

import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy import ndimage

# #####################################
# Reading and Saving files
# #####################################

def ReadFile(filepath, verbose=1):
    if verbose == 1:
        print("Reading:", filepath)

    filename = os.path.basename(filepath)

    if ".png" in filename: img = Read_png_file(filepath)
    if ".nii" in filename: img = Read_nifti_file(filepath)

    return img

def Read_png_file(filepath):
    img = imageio.imread(filepath)
    return img

def Read_nifti_file(filepath):
    """Read and load nifti file"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


# #####################################
# Pre-process functions
# #####################################

def Normalize(input_file):
    """Normalize the input_file"""
    min = np.min(input_file)
    max = np.max(input_file)
    input_file[input_file < min] = min
    input_file[input_file > max] = max
    if min != max:
        input_file = (input_file - min) / (max - min)
    input_file = np.array(input_file)
    return input_file

def Resize_2D(img, desired_width, desired_height):
    """Resize 2D image"""
    # Get current depth
    current_width = np.shape(img)[0]
    current_height = np.shape(img)[1]
    # Compute depth factor
    width = current_width / desired_width
    height = current_height / desired_height
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (width_factor, height_factor), order=1)
    return img


# #####################################
# Loading data
# #####################################

def Array_2_5D(file_path, paths, width, height, neighborhood, label):
    neighbors = int((neighborhood-1)/2)

    if not label:
        fname = os.path.basename(file_path)
        fdir  = os.path.dirname(file_path)

        numberSlice = re.split('_|\.', fname)[1]

        neigh_paths, input_file = [], []
        for slice in range(-neighbors,neighbors+1):
            indexNumberSlice = fname.rfind(numberSlice)
            new_fname = fname[:indexNumberSlice] + fname[indexNumberSlice:].replace(numberSlice, str(int(numberSlice)+slice))

            new_path = os.path.join(fdir,new_fname)
            if new_path not in paths:
                fake_slice = np.zeros((width, height))
                input_file.append(fake_slice)

            else:
                input_file.append(Process_data(new_path, label))

        input_file = np.array(input_file)
        input_file = np.transpose(input_file)

    else:
        input_file = np.array(Process_data(file_path, label))

    return input_file

def Process_data(path, label):
    """Load input file"""
    # Read file
    input_file = ReadFile(path, verbose=0)
    # Normalize
    input_file = Normalize(input_file)
    if label:
        input_file[input_file<0.5]=0.0
        input_file[input_file>=0.5]=1.0

    return input_file


# #####################################
# Display functions
# #####################################

def Plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of CT slices"""
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

























