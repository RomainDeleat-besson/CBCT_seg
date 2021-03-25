import os
import re

import imageio
import itk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage import exposure
import sys
# np.set_printoptions(threshold=sys.maxsize)

# #####################################
# Reading and Saving files
# #####################################

def ReadFile(filepath, verbose=1):
    if verbose == 1:
        print("Reading:", filepath)
    
    scan = itk.imread(filepath)
    scan = itk.GetArrayFromImage(scan)
    return scan

#     filename = os.path.basename(filepath)

#     if ".png" in filename: img = Read_png_file(filepath)
#     if ".nii" in filename: img = Read_nifti_file(filepath)

#     return img

# def Read_png_file(filepath):
#     img = imageio.imread(filepath)
#     return img

# def Read_nifti_file(filepath):
#     """Read and load nifti file"""
#     # Read file
#     scan = nib.load(filepath)
#     # Get raw data
#     scan = scan.get_fdata()
#     return scan


def SaveFile(filepath, data, ext, verbose=1):
    if verbose == 1:
        print("Saving:", filepath)

    if ext == ".png": Save_png(filepath, data)

def Save_png(filepath, data):
    imageio.imsave(filepath, data)    


# #####################################
# Pre-process functions
# #####################################

def Normalize(input_file,in_min=None,in_max=None,out_min=0,out_max=1):
    """Normalize the input_file"""
    if in_min is None:
        in_min = input_file.min()
    if in_max is None:
        in_max = input_file.max()
    input_file = exposure.rescale_intensity(input_file, in_range=(in_min, in_max), out_range=(out_min,out_max))
    return input_file

def Adjust_Contrast(input,out_min=None,out_max=None,pmin=10,pmax=90):

    if out_min is None:
        out_min = input.min()
    if out_max is None:
        out_max = input.max()

    val_min, val_max = np.percentile(input[input!=0], (pmin, pmax))
    print(val_min, val_max)
    input = Normalize(input, val_min,val_max, out_min,out_max)

    input = exposure.equalize_hist(input)
    print(input.dtype, input.min(), input.max())

    input = Normalize(input, out_min=out_min, out_max=out_max)

    return input

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

def Deconstruction(img, filename, outdir, desired_width=512, desired_height=512):
    """Separate each slice of a 3D image"""
    for z in range(img.shape[0]):
        slice = img[z,:,:]
        out = outdir+'/'+os.path.basename(filename).split('.')[0]+'_'+str(z)+'.png'
        slice = Resize_2D(slice, desired_width, desired_height)
        imageio.imwrite(out, slice.astype(np.uint8))


# #####################################
# Loading data
# #####################################

def Array_2_5D(file_path, paths, width, height, neighborhood, label):
    neighbors = int((neighborhood-1)/2)

    if not label:
        fname = os.path.basename(file_path)
        fdir  = os.path.dirname(file_path)

        numberSlice = re.split('_|\.', fname)[-2]

        neigh_paths, input_file = [], []
        for slice in range(-neighbors,neighbors+1):
            indexNumberSlice = fname.rfind(numberSlice)
            new_fname = fname[:indexNumberSlice] + fname[indexNumberSlice:].replace(numberSlice, str(int(numberSlice)+slice))

            new_path = os.path.join(fdir,new_fname)
            if new_path not in paths:
                fake_slice = np.zeros((width, height), dtype=np.float32)
                input_file.append(fake_slice)

            else:
                # Read file
                File = ReadFile(path, verbose=0)
                # Normalize
                File = Normalize(File)
                input_file.append(File)

        input_file = np.array(input_file, dtype=np.float32)
        input_file = np.transpose(input_file)

    else:
        # Read file
        File = ReadFile(path, verbose=0)
        # Normalize
        File = Normalize(File)
        input_file = np.array(File, dtype=np.float32)
        # // verifier si les labels ont la valeur 1 ou 0
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


# #####################################
# Post-process functions
# #####################################

def Reconstruction(dir, outdir):
    """Reconstruction of a 3D scan from the 2D slices"""
























