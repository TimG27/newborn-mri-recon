"""
This file defines functions that add phases to each of the 60 adult mri images.
"""

import cmath
import random
import SimpleITK as sitk
from preproccessing.utils.coil_utils import *


def random_add_phase(folder1_path, folder2_path, destination_path, trans_path):
    """
    This function will perform the phase addition. It takes every file in folder1 (imageA) and matches it randomly to a
    file in folder 2 (imageB). It will then execute the registration, resampling, and complex number using these
    two files. All the complex valued files are stored in the destination path.
    Args:
        destination_path: Path where the complex valued arrays can be stored
        folder1_path: Path that contains the folder with original dataset (60 adult MRI images)
        folder2_path: Path that contains the folder with aux dataset (19 complex valued images)
        trans_path: path to save the transforms
    Returns:

    """
    random.seed(227)  # Ensure reproducibility

    files_folder1 = os.listdir(folder1_path)
    files_folder2 = os.listdir(folder2_path)

    MRI_files = []
    Ref_files = []

    for file_folder1 in files_folder1:
        file_path_folder1 = os.path.join(folder1_path, file_folder1)
        image_A = np.load(file_path_folder1)

        file_folder2 = random.choice(files_folder2)
        file_path_folder2 = os.path.join(folder2_path, file_folder2)
        image_B = np.load(file_path_folder2)

        MRI_files.append(file_folder1)
        Ref_files.append(file_folder2)

        # PREREGISTRATION ALIGNMENT

        image_A = pre_reg_MRI(image_A)
        image_B = pre_reg_Ref(image_B)

        try:
            final_transform, image_a, image_b_magnitude, image_b_phase = register_image(image_A, image_B)

            sitk.WriteTransform(final_transform, str(trans_path + file_folder1 + '.tfm'))

            fixed_image, image_b_aligned, image_b_aligned_phase = resample_img(final_transform, image_a,
                                                                               image_b_magnitude, image_b_phase)
            
            complex_array = combine_to_complex(fixed_image, image_b_aligned_phase)

            np.save(destination_path + '/' + file_folder1 + '.npy', complex_array)

        except Exception as e:
            print(f"Error processing files '{file_folder1}' and '{file_folder2}': {str(e)}")

    mapping = list(zip(MRI_files, Ref_files))

    csv_file = '../file_mapping.csv'

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(mapping)


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f}"
    )


def register_image(fixed_image, moving_image):
    """
    This function will register a moving image to a fixed image.

    Args:
        fixed_image (numpy ndarray) : This is the original image (from the 60 adult MRI dataset)
        moving_image (numpy ndarray) : This is one of the complex images (from the 19 adult MRI dataset)

    Returns: The registered transformation

    """
    image_a = sitk.GetImageFromArray(fixed_image)
    image_a = sitk.Cast(image_a, sitk.sitkFloat32)

    image_b = moving_image.astype(np.complex64)
    image_b_magnitude = np.abs(image_b)
    image_b_phase = np.angle(image_b)

    image_b_magnitude = sitk.GetImageFromArray(image_b_magnitude)
    image_b_magnitude = sitk.Cast(image_b_magnitude, sitk.sitkFloat32)

    fixed_image = image_a
    moving_image = image_b_magnitude

    # New: BSplice registration instead of Euler's 3D registration
    transformDomainMeshSize = [2] * moving_image.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_image, transformDomainMeshSize)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()

    # Faster optimiser
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1,
        minStep=1e-4,
        numberOfIterations=100,
    )

    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    try:
        final_transform = R.Execute(fixed_image, moving_image)
        print('Registration success!')
    except Exception as e:
        print("Registration failed. Please check data formats: ", e)

    return final_transform, image_a, image_b_magnitude, image_b_phase


def resample_img(transform, fixed_mag, moving_mag, moving_phase):
    """
    This function resamples the moving image maps based on the transform that was determined during registration
    Args:
        transform: The transformation from registration
        fixed_mag: The magnitude map from the fixed image
        moving_mag: The magnitude map from the moving image (will be resampled)
        moving_phase: The phase map from the moving image (will be resampled)

    Returns: The aligned fixed image (magnitude), aligned image (mag), aligned image (phase)

    """
    fixed_image = fixed_mag
    final_transform = transform
    moving_image = moving_mag
    image_b_phase = moving_phase

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)

    try:
        image_b_aligned = resampler.Execute(moving_image)
        print("Magnitude alignment success!", sitk.GetArrayFromImage(image_b_aligned).shape)
    except Exception as e:
        print("Resampling failed. Please check data formats: ", e)

    image_b_phase = sitk.GetImageFromArray(image_b_phase)
    image_b_phase = sitk.Cast(image_b_phase, sitk.sitkFloat32)

    try:
        image_b_aligned_phase = resampler.Execute(image_b_phase)
        print("Phase alignment success!", sitk.GetArrayFromImage(image_b_aligned_phase).shape)
    except Exception as e:
        print("Resampling failed. Please check data formats: ", e)

    if sitk.GetArrayFromImage(fixed_image).shape == sitk.GetArrayFromImage(image_b_phase).shape:
        print("Dimensions Match.")

    return fixed_image, image_b_aligned, image_b_aligned_phase


def combine_to_complex(magnitude_image, phase_image):
    """
    This function combines a magnitude and phase image to one complex format image
    Args:
        magnitude_image: Magnitude-only numpy array
        phase_image: Phase-only numpy array

    Returns: Complex image array

    """
    image_a = magnitude_image
    image_b_aligned_phase = phase_image

    magnitudes = sitk.GetArrayFromImage(image_a)
    phases = sitk.GetArrayFromImage(image_b_aligned_phase)

    vectorized_rect = np.vectorize(cmath.rect)
    complex_array = vectorized_rect(magnitudes, np.deg2rad(phases))

    print('Complex array, of dimensions ', complex_array.shape)

    return complex_array
