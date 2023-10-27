"""
This file defines functions that are used to generate the under sampling poisson masks.
"""

import os
import random
from typing import Tuple, List

import numpy as np


def poisson_disc2d(pattern_shape: Tuple[int], k: int, r: float) -> np.ndarray:
    """Return poisson_disc2d undersampling pattern.

    Implementation of the Poisson disc sampling available at:
    https://scipython.com/blog/poisson-disc-sampling-in-python/

    :param pattern_shape: shape of the desired sampling pattern.
    :param k: Number of points to sample around each reference point as candidates for new sample points.
    :param r: Minimum distance between samples.
    :return: Boolean mask for under sampling k-space.
    """
    pattern_shape = (pattern_shape[0] - 1, pattern_shape[1] - 1)
    width, height = pattern_shape
    # Cell side length
    a = r / np.sqrt(2)
    # Number of cells in the x- and y-directions of the grid
    nx, ny = int(width / a) + 1, int(height / a) + 1

    # A list of coordinates in the grid of cells
    coords_list = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    # Initilalize the dictionary of cells: each key is a cell's coordinates, the
    # corresponding value is the index of that cell's point's coordinates in the
    # samples list (or None if the cell is empty).
    cells = {coords: None for coords in coords_list}

    def get_cell_coords(pt: Tuple[int]) -> Tuple[int]:
        """Get the coordinates of the cell that pt = (x,y) falls in."""
        return int(pt[0] // a), int(pt[1] // a)

    def get_neighbours(coords: Tuple[int]) -> List[int]:
        """
        Return the indexes of points in cells neighbouring cell at coords.
        For the cell at coords = (x,y), return the indexes of points in
        the cells with neighbouring coordinates illustrated below: ie
        those cells that could contain points closer than r.

                             ooo
                            ooooo
                            ooXoo
                            ooooo
                             ooo

        :param coords: Cell coordinates.
        :return: List of indexes of points in neighbouring cells.
        """
        dxdy = [
            (-1, -2),
            (0, -2),
            (1, -2),
            (-2, -1),
            (-1, -1),
            (0, -1),
            (1, -1),
            (2, -1),
            (-2, 0),
            (-1, 0),
            (1, 0),
            (2, 0),
            (-2, 1),
            (-1, 1),
            (0, 1),
            (1, 1),
            (2, 1),
            (-1, 2),
            (0, 2),
            (1, 2),
            (0, 0),
        ]

        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < nx and 0 <= neighbour_coords[1] < ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store this index of the contained point.
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(pt: Tuple[int]) -> bool:
        """Is pt a valid point to emit as a sample? It must be no closer than r from any other point:
        check the cells in its immediate neighbourhood.

        :param pt: Point coordinates
        :return: True if valid, False otherwise.
        """
        cell_coords = get_cell_coords(pt)
        for idx in get_neighbours(cell_coords):
            nearby_pt = samples[idx]
            # Squared distance between or candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0] - pt[0]) ** 2 + (nearby_pt[1] - pt[1]) ** 2
            if distance2 < r ** 2:
                # The points are too close, so pt is not a candidate.
                return False

        # All points tested: if we're here, pt is valid
        return True

    def get_point(k: int, refpt: Tuple[int]) -> Tuple[int]:
        """Try to find a candidate point relative to refpt to emit in the sample. We draw up to k points from the
        annulus of inner radius r, outer radius 2r around the reference point, refpt. If none of them are suitable
        (because they're too close to existing points in the sample), return False. Otherwise, return the pt.

        :param k: Number of points to sample around each ref point.
        :param refpt: Coords of ref point.
        :return: Point if valid, else False.
        """
        i = 0
        while i < k:
            rho, theta = np.random.uniform(r, 2 * r), np.random.uniform(0, 2 * np.pi)
            pt = refpt[0] + rho * np.cos(theta), refpt[1] + rho * np.sin(theta)
            if not (0 < pt[0] < width and 0 < pt[1] < height):
                # This point falls outside the domain, so try again.
                continue
            if point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    # Pick a random point to start with.
    pt = (np.random.uniform(0, width), np.random.uniform(0, height))
    samples = [pt]
    # Our first sample is indexed at 0 in the samples list...
    cells[get_cell_coords(pt)] = 0
    # ... and it is active, in the sense that we're going to look for more points
    # in its neighbourhood.
    active = [0]

    nsamples = 1
    # As long as there are points in the active list, keep trying to find samples.
    while active:
        # choose a random "reference" point from the active list.
        idx = np.random.choice(active)
        refpt = samples[idx]
        # Try to pick a new point relative to the reference point.
        pt = get_point(k, refpt)
        if pt:
            # Point pt is valid: add it to the samples list and mark it as active
            samples.append(pt)
            nsamples += 1
            active.append(len(samples) - 1)
            cells[get_cell_coords(pt)] = len(samples) - 1
        else:
            # We had to give up looking for valid points near refpt, so remove it
            # from the list of "active" points.
            active.remove(idx)
    samples = np.rint(np.array(samples)).astype(int)
    samples = np.unique(samples[:, 0] + 1j * samples[:, 1])
    samples = np.column_stack((samples.real, samples.imag)).astype(int)
    poisson_pattern = np.zeros((pattern_shape[0] + 1, pattern_shape[1] + 1), dtype=bool)
    poisson_pattern[samples[:, 0], samples[:, 1]] = True
    return poisson_pattern


def centered_circle(image_shape: Tuple[int], radius: float) -> np.ndarray:
    """Creates a boolean centered circle image with a pre-defined radius

    :param image_shape: shape of the desired image
    :param radius: radius of the desired circle
    :return: circle image. It is a boolean image
    """
    center_x = int((image_shape[0] - 1) / 2)
    center_y = int((image_shape[1] - 1) / 2)

    X, Y = np.indices(image_shape)
    circle_image = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius ** 2  # type: bool

    return circle_image


def poisson_disc_pattern(
        pattern_shape: Tuple[int], center: bool = True, radius: int = 5, k: int = 5, r: int = 2
) -> np.ndarray:
    """Creates a uniformly distributed sampling pattern.

    :param pattern_shape: shape of the desired sampling pattern.
    :param center: boolean variable telling whether or not sample low frequencies. Defaults to True.
    :param radius: variable telling radius (2D) to be sampled in the centre. Defaults to 5.
    :param k: Number of points around each reference point as candidates for a new sample point. Defaults to 5.
    :param r: Minimum distance between samples. Defaults to 2.
    :return: sampling pattern. It is a boolean image
    """
    if center is False:
        return poisson_disc2d(pattern_shape, k, r)
    else:
        pattern1 = poisson_disc2d(pattern_shape, k, r)
        pattern2 = centered_circle(pattern_shape, radius)
        return np.logical_or(pattern1, pattern2)


def generate_masks(shape, destin_directory, count=100):
    counter = 0

    while counter < count:

        image = poisson_disc_pattern(shape, True, 30, 20, 2)
        non_zero_count = np.count_nonzero(image)
        total_elements = image.size
        non_zero_percentage = non_zero_count / total_elements
        print(non_zero_count, total_elements, non_zero_percentage)

        if 0.1999 <= non_zero_percentage <= 0.2001:
            print('Saved')
            filename = f"mask_p_{counter}.npy"

            save_path = os.path.join(destin_directory, filename)

            np.save(save_path, image)
            counter += 1

            # plt.imshow(image, cmap='gray')
            # plt.show()


def undersamp(image_path, mask_path, destin_path):
    """
    For each image in the image path, a random mask will be picked from the mask path and multiplied with it.
    :param image_path: path with images (now 4D coil acquisitions)
    :param mask_path: path with 100 masks generated using poisson disc patterns
    :param destin_path: path to save the undersampled images
    """

    image_files = os.listdir(image_path)
    mask_files = os.listdir(mask_path)

    for image_file in image_files:
        image_file_path = os.path.join(image_path, image_file)
        mask_file_path = os.path.join(mask_path, random.choice(mask_files))

        image_array = np.load(image_file_path)  # (12, 200, 256, 216)
        mask_array = np.load(mask_file_path)  # (200, 256)

        coils = np.moveaxis(image_array, 0, 0)  # unstacking

        result_array = []

        for acq in coils:
            undersp_array = np.multiply(mask_array[..., np.newaxis], acq)
            print(undersp_array.shape)
            result_array.append(undersp_array)

        result_array = np.stack(result_array)
        print('result', result_array.shape)

        destination_file = os.path.join(destin_path, image_file)

        np.save(destination_file, result_array)
