"""
This file contains the code that was used to separate the 3D image arrays into 2D slices for the model to train.
"""

import os
import numpy as np
import pandas as pd

# Set the paths for the input and output directories
input_directory = '/home/timothy2/projects/def-rmsouza/timothy2/mini_mri_dataset/coiled_adult_60'
output_directory = "/home/timothy2/scratch/cascade-try/slicesv3"
os.makedirs(output_directory, exist_ok=True)

csv_data = []

# Iterate over each 3D numpy array file
file_list = sorted(os.listdir(input_directory))
for i, file_name in enumerate(file_list):
    file_path = os.path.join(input_directory, file_name)
    if file_path.endswith('.npy'):
        array_3d = np.load(file_path)

        # Split the 3D array into 2D arrays
        num_slices = array_3d.shape[3]
        for index in range(num_slices):
            slice_2d = array_3d[:, :, :, index]

            # Save the 2D slice as a separate file
            slice_filename = f'{file_name}_{index}.npy'
            slice_filepath = os.path.join(output_directory, slice_filename)
            np.save(slice_filepath, slice_2d)

            # Update the CSV data
            split = 'train' if i < 40 else 'valid' if i < 50 else 'test'
            csv_data.append({
                'espirit_path': slice_filepath,
                'file_name': file_name,
                'index': index,
                'split': split
            })

# Save the CSV file
csv_df = pd.DataFrame(csv_data)
csv_filepath = '/home/timothy2/scratch/cascade-try/slice_ids_v5.csv'
csv_df.to_csv(csv_filepath, index=False)
