import glob
import gzip
import os
import random
import re
import sys

import imageio
import itk
import matplotlib.pyplot as plt
import medpy.io
import nibabel as nib
import nrrd
import numpy as np
import skimage.transform as sktf
import tensorflow as tf
from scipy import ndimage
from skimage import exposure, io
from tensorflow.keras.preprocessing.image import save_img


# #####################################
# Reading files
# #####################################

def ReadFile(filepath, verbose=1):
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
    return img, header

def Read_gipl(filepath):
    img, header = medpy.io.load(filepath)
    return img, header

def Read_nrrd(filepath):
    img, header = nrrd.read(filepath)
    return img, header

def Read_png(filepath):
    img = io.imread(filepath)
    return img


# #####################################
# Saving files
# #####################################

def SaveFile(filepath, data, header=None, verbose=1):
    if verbose == 1:
        print("Saving:", filepath)

    ext = os.path.basename(filepath)
    filepath = filepath.replace('.gz','')

    # if type(data).__module__ == np.__name__: 
    #     if ImageType is None: data = itk.GetImageFromArray(data)
    #     else: data = itk.PyBuffer[ImageType].GetImageFromArray(data)

    # if ImageType is None: writer = itk.ImageFileWriter.New()
    # else: writer = itk.ImageFileWriter[ImageType].New(FileName=filepath, Input=data)
    # writer.Update()

    if ".png" in ext: Save_png(filepath, data)
    if ".nii" in ext: Save_nifti(filepath, data, header)
    if ".gipl" in ext: Save_gipl(filepath, data, header)
    if ".nrrd" in ext: Save_nrrd(filepath, data, header)
    if ".gz" in ext: Save_gz(filepath, data)

def Save_nifti(filepath, data, header):
    data = nib.nifti1.Nifti1Image(data, None, header=header)
    nib.nifti1.save(data, filepath)

def Save_gipl(filepath, data, header):
    # from medpy.io import save
    medpy.io.save(data, filepath, header)

def Save_nrrd(filepath, data, header):
    nrrd.write(filepath, data, header)

def Save_png(filepath, data):
    if data.ndim==2:
        data = np.reshape(data, (data.shape[0],data.shape[1],1))
    save_img(filepath, data)

def Save_gz(filepath, data):
    with open(filepath, 'rb') as f_in, gzip.open(filepath+'.gz', 'wb') as f_out:
        f_out.writelines(f_in)
    os.remove(filepath)


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
    # print(img.shape)
    for z in range(img.shape[2]):
        slice = img[:,:,z]
        out = outdir+'/'+os.path.basename(filename).split('.')[0]+'_'+str(z)+'.png'
        slice = Resize_2D(slice, desired_width, desired_height)

        # ImageType = itk.Image[itk.UC,2]
        # slice = slice.astype(np.ubyte)
        # slice = itk.PyBuffer[ImageType].GetImageFromArray(slice)
        # slice = itk.GetImageFromArray(slice)
        SaveFile(out, slice, verbose=0)
        # SaveFile(out, slice.astype(np.uint8), verbose=0)


# #####################################
# Loading data
# #####################################

def Array_2_5D(file_path, paths, width, height, neighborhood, label):
    neighbors = int((neighborhood-1)/2)

    if not label:
    #     fname = os.path.basename(file_path)
    #     fdir  = os.path.dirname(file_path)

    #     numberSlice = re.split('_|\.', fname)[-2]
        
        
    #     # print("==================================================")
    #     # print("file path:", file_path)

    #     neigh_paths, input_file = [], []
    #     for slice in range(-neighbors,neighbors+1):
    #         # print("slice:", slice)
    #         indexNumberSlice = fname.rfind(numberSlice)
    #         new_fname = fname[:indexNumberSlice] + fname[indexNumberSlice:].replace(numberSlice, str(int(numberSlice)+slice))

    #         new_path = os.path.join(fdir,new_fname)
    #         # print("new path:", new_path)
    #         if new_path not in paths:
    #             fake_slice = np.zeros((width, height), dtype=np.float32)
    #             input_file.append(fake_slice)

    #         else:
    #             # Read file
    #             File = ReadFile(file_path, array=True, verbose=0)
    #             # Normalize
    #             File = Normalize(File, out_min=0,out_max=1)
    #             input_file.append(File)

        # Read file
        File, _ = ReadFile(file_path, verbose=0)

        if np.amax(File)>0:
            File = (File-np.amin(File))/(np.amax(File)-np.amin(File))
            
        File = np.float32(np.uint8(255*File))/255

        input_file = np.array(File, dtype=np.float32)
        
    else:
        File, _ = ReadFile(file_path, verbose=0)
        
        File[File<127.5]=0
        File[File>=127.5]=1
                
        input_file = np.array(File, dtype=np.float32)
                
    return input_file

# #####################################
# Data Aug & Dataset
# #####################################

def _rotate (img, k):
    return sktf.rotate(img, angle=k, mode='constant', cval=0.0, resize=False)

def _shift (img, k, axis): # shape: (H,W,C)
    x = np.roll(img, k, axis)
    if k > 0 and axis == 0: x[:k,:,:] = 0
    if k < 0 and axis == 0: x[k:,:,:] = 0
    if k > 0 and axis == 1: x[:,:k,:] = 0
    if k < 0 and axis == 1: x[:,k:,:] = 0
    return x

def _shear (img, k):
    return sktf.warp(img, sktf.AffineTransform(shear=k), order=1, preserve_range=True, mode='constant', cval=0.0)

def _centerZoom (img, k):
    return sktf.resize(img[k:-k, k:-k, :], img.shape, anti_aliasing=False)

def map_decorator(func):
    def wrapper(*args):
        return tf.py_function(
            func=func,
            inp=[*args],
            Tout=[a.dtype for a in args])
    return wrapper

def aug_layers (x, seed_rot, seed_shift0, seed_shift1, seed_shear, seed_zoom):
    x = x.numpy()
    x = _rotate(x, seed_rot)           # Angle in degrees
    x = _shift(x, seed_shift0, axis=0) # Shift in pixels
    x = _shift(x, seed_shift1, axis=1) # Shift in pixels
    x = _shear(x, seed_shear)          # Recommended range: [-0.2, 0.2]
    x = _centerZoom(x, seed_zoom+1)    # Zoom in pixels (+1 to avoid empty array if k=0)

    if label:
        x[x<0.5] = 0
        x[x>=0.5] = 1
    else:
        x = np.float32(np.uint8(255*x))/255

    return x
    
@map_decorator
def augment(x, y):
    # random module much faster than numpy
    seed_rot = int(random.uniform(0, 360))
    seed_shift0 = int(random.uniform(-30, 30))
    seed_shift1 = int(random.uniform(-30, 30))
    seed_shear = random.uniform(-0.2, 0.2)
    seed_zoom = int(random.uniform(0, 15))
#     print(seed)
#     np.random.seed(seed)

    x = aug_layers(x, seed_rot, seed_shift0, seed_shift1, seed_shear, seed_zoom)
    y = aug_layers(y, seed_rot, seed_shift0, seed_shift1, seed_shear, seed_zoom)
    return x, y

@map_decorator
def augment_heat_map(y):
    # random module much faster than numpy
    seed_rot = int(random.uniform(0, 360))
    seed_shift0 = int(random.uniform(-30, 30))
    seed_shift1 = int(random.uniform(-30, 30))
    seed_shear = random.uniform(-0.2, 0.2)
    seed_zoom = int(random.uniform(0, 15))

    y = aug_layers(y, seed_rot, seed_shift0, seed_shift1, seed_shear, seed_zoom)
    return y

def create_dataset(x, y, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)#tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.shuffle(32*BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


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

