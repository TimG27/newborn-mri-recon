# MRI Reconstruction with Deep Learning

## Overview

This repository contains code and resources for a deep learning project focused on MRI reconstruction. The project includes the training of a deep learning model using complex 12-coil adult MRI images and testing its domain adaptivity on newborn MRI images, including the evaluation of contrast inversion techniques.

## Project Structure

- `preprocessing/`: Contains scripts related to data preprocessing before training.
- `unet/`: Code for the UNet model tested in this project.
- `hybrid-cascade/`: Code for the hybrid cascade model.
- `hybrid-cascade-contrast-inversion/`: Extensions for the hybrid cascade model with contrast inversion.

## Project Flow

1. **Data Preprocessing**: 
    - Directory: `preprocessing/`
    - Description: In this step, we prepare and preprocess the MRI data.

    a. **Incorporating a phase component into the magnitude-only images and converting them into 12 coil configurations**:
      - File(s): `adding_phase_and_coil.py`
      - Description: Every magnitude-only image is randomly matched with a complex-valued MRI image from an external dataset. After image registration, the phase component is extracted from the complex image and incorporated into the magnitude image, thus transforming it into a complex image.
    b. **Generating 2D MRI slices**:
      - File(s): `slicer.py`
      - Description: Each 3D MRI volume is sliced into 2D images for training.
    c. **Generating Undersampling masks**:
      - File(s): `undersampling_masks.py`
      - Description: Poisson disks are utilized to generate undersampling masks with an undersampling factor, R = 5.

2. **UNet Model Training**:
   - Directory: `unet/`
   - Description: The UNet model is trained on the adult MRI images.
   
   a. **Training**:
      - File(s): `training-adult.py`
      - Description: Using a custom written data loader, the images are fed into the unet model for training.
   b. **Testing**:
      - File(s): `testing-newborns.py`
      - Description: The model is tested on a dataset of newborn images

3. **Hybrid Cascade Model Training**:
   - Directory: `hybrid-cascade/`
   - Description: The hybrid cascade model, without any domain adaptation techniques, is trained.
     
   a. **Training**:
      - File(s): `training-adult.py`
      - Description: The hybrid cascade model is trained.
   b. **Testing**:
      - File(s): `testing-newborns.py`
      - Description: The model is tested on a dataset of newborn images

4. **Testing with Contrast Inversion**:
   - Directory: `hybrid-cascade-contrast-inversion/`
   - Description: The hybrid cascade model is trained on adult MR images that have undergone contrast inversion.
  
   a. **Training**:
      - File(s): `training-adult.py`
      - Description: The hybrid cascade model is trained.
   b. **Testing**:
      - File(s): `testing-newborns.py`
      - Description: The model is tested on a dataset of newborn images

