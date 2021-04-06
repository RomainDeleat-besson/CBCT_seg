import glob
import gzip
import os
import re
import sys

import imageio
import itk
import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
import numpy as np
import tensorflow as tf
from scipy import ndimage
from skimage import exposure, io
from tensorflow.keras.preprocessing.image import save_img
# np.set_printoptions(threshold=sys.maxsize)



# #####################################
# Reading files
# #####################################

def ReadFile(filepath, ImageType=None, verbose=1):
    if verbose == 1:
        print("Reading:", filepath)
    
    # img = itk.imread(filepath)

    # if ImageType is None: reader = itk.ImageFileReader.New(FileName=filepath)
    # else: reader = itk.ImageFileReader[ImageType].New(FileName=filepath)
    # reader.Update()
    # img = reader.GetOutput()

    # if array: img = itk.GetArrayFromImage(img)

    header = None

    if '.nii' in filepath: img, header = Read_nifti(filepath)
    if '.gipl' in filepath: img, header = Read_gipl(filepath)
    if '.nrrd' in filepath: img, header = Read_nrrd(filepath)
    if '.png' in filepath: img = Read_png(filepath)

    return img, header

def Read_nifti(filepath):
    img = nib.load(filepath)
    header = img.header
    img = img.get_fdata()
    # qform = header.get_qform(coded=False)
    # print(qform)
    # qform[0,0] = -qform[0,0]
    # qform[1,1] = -qform[1,1]
    # print(qform)
    # header.set_qform(qform, code='mni')
    return img, header

def Read_gipl(filepath):
    # img, header = medpy.io.load(filepath)
    # return img, header
    print()

def Read_nrrd(filepath):
    img, header = nrrd.read(filepath) #, index_order='F'
    return img, header

def Read_png(filepath):
    img = io.imread(filepath)
    return img


# #####################################
# Saving files
# #####################################

def SaveFile(filepath, data, ImageType=None, verbose=1):
    if verbose == 1:
        print("Saving:", filepath)

    ext = os.path.basename(filepath)
    filepath = filepath.replace('.gz','')

    # if type(data).__module__ == np.__name__: 
    #     if ImageType is None: data = itk.GetImageFromArray(data)
    #     else: data = itk.PyBuffer[ImageType].GetImageFromArray(data)

    # if ImageType is None: writer = itk.ImageFileWriter.New()
    # else: writer = itk.ImageFileWriter[ImageType].New()
    # writer.SetFileName(filepath)
    # writer.SetInput(data)
    # writer.Update()

    if ".png" in ext: Save_png(filepath, data)
    if ".nrrd" in ext: Save_nrrd(filepath, data)
    if ".gz" in ext: Save_gz(filepath, data)


def Save_nifti(data, filepath, verbose=1):
    if verbose == 1:
        print("Saving:", filepath)
    nib.nifti1.save(data, filepath)

def Save_png(filepath, data):
    if data.ndim==2:
        data = np.reshape(data, (data.shape[0],data.shape[1],1))
    save_img(filepath, data)
    # imageio.imsave(filepath, data)

def Save_nrrd(filepath, data):
    nrrd.write(filepath, data)
    # ImageType = itk.Image[itk.US, 3]
    # writer = itk.ImageFileWriter[ImageType].New(FileName=filepath.replace('.gz',''), Input=data)
    # writer.Update()

def Save_gz(filepath, data):
    with open(filepath.replace('.gz', ''), 'rb') as f_in, gzip.open(filepath, 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(filepath.replace('.gz', ''))


# #####################################
# Pre-process functions
# #####################################

def Normalize(input_file,in_min=None,in_max=None,out_min=0,out_max=255):
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
    # print(val_min, val_max)
    input = Normalize(input, val_min,val_max, out_min,out_max)

    input = exposure.equalize_hist(input)
    # print(input.dtype, input.min(), input.max())

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
    print(img.shape)
    for z in range(img.shape[2]):
        slice = img[:,:,z]
        out = outdir+'/'+os.path.basename(filename).split('.')[0]+'_'+str(z)+'.png'
        slice = Resize_2D(slice, desired_width, desired_height)

        ImageType = itk.Image[itk.UC,2]
        slice = slice.astype(np.ubyte)
        # slice = itk.PyBuffer[ImageType].GetImageFromArray(slice)
        # slice = itk.GetImageFromArray(slice)
        SaveFile(out, slice, ImageType, verbose=0)
        # SaveFile(out, slice.astype(np.uint8), verbose=0)


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
                File = ReadFile(file_path, verbose=0)
                # Normalize
                File = Normalize(File)
                input_file.append(File)

        input_file = np.array(input_file, dtype=np.float32)
        input_file = np.reshape(input_file, (width, height, neighborhood))

    else:
        # Read file
        File = ReadFile(file_path, verbose=0)
        # Normalize
        File = Normalize(File)
        File[File<(np.max(File))/2.0]=0
        File[File>=(np.max(File))/2.0]=1
        input_file = np.array(File, dtype=np.float32)

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

def Reconstruction(filename, dir, original_img, outdir):
    """Reconstruction of a 3D scan from the 2D slices"""
    size = original_img.shape
    print(size)
    img = np.zeros(size)
    normpath = os.path.normpath('/'.join([dir,filename+'*']))
    for slice_path in glob.iglob(normpath):
        slice, header = ReadFile(slice_path, verbose=0)
        slice = Resize_2D(slice, size[0], size[1])
        slice_nbr = int(re.split('_|\.',slice_path)[-2])
        img[:,:,slice_nbr] = slice
    return img
























