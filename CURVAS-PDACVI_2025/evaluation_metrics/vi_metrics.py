import numpy as np
from scipy.ndimage import binary_erosion, convolve, binary_dilation


def get_involvement(tumor: np.array, vessel: np.array, vessel_value: int, image: np.array=None) -> dict:
    """
    Compute the aggregated values of the Vascular Involvement (VI) for each structure

    Parameters
    ----------
    tumor: np.array
        Mask of the PDAC.
    vessel: np.array
        Mask of the vessel strucutres.
    vessel_value: int
        Value indicating which vascular structure to assess.
    image: np.array
        CT image array.

    Returns
    -------
    aggregated_metrics: dictionary
        Dictionary of Wasserstein metrics for each vascular structure.
    """

    max_degrees = {'cor': 0, 'sag': 0, 'ax': 0}
    planes = {'cor': 0, 'sag': 1, 'ax': 2}

    for plane in planes:
        merged_masks = np.where(vessel==1, 2, tumor)
        slice_list = np.unique(sorted(np.where(merged_masks)[planes[plane]]))

        for slice_id in slice_list:
            slice_tumor = get_plane(plane, tumor, slice_id)
            if 1 not in slice_tumor: continue

            slice_vessels = get_plane(plane, vessel, slice_id)
            slice_vessel = get_selected_segments(slice_vessels, vessel_value)
            if 1 not in slice_vessel: continue

            cropped_slice_tumor, cropped_slice_vessel, _ = crop_slices_together(slice_tumor, slice_vessel)
            _, _, tumor_contact, vessel_edge = assess_vascular_involvement(cropped_slice_tumor, cropped_slice_vessel, dilation_margin=3)

            degrees = (tumor_contact / np.sum(vessel_edge == 1) * 360) if np.sum(vessel_edge == 1) > 0 else 0
            
            if max_degrees[plane] < degrees: max_degrees[plane] = degrees

    return max_degrees


def assess_vascular_involvement(lesion_mask, vessel_mask, dilation_margin=1) -> tuple[np.array, int, np.array]:
    """
    Assess direct overlap and proximity-based vessel involvement.

    Parameters
    ----------
        lesion_mask: np.ndarray
            Binary mask with 1 = lesion.
        vessel_mask: np.ndarray)
            Binary mask with 1 = vessel.
        dilation_margin: int
            Number of pixels to dilate lesion mask.

    Returns
    -------
        merged: np.array
            Array containing lesion (=1), vessel edge wo contact (=2), vessel edge in contact (=3).
        tumor_contact: int
            Number of pixels of the vessel that are invaded by the tumor.
        vessel_edge: np.array
            Binary mask with 1 = vessel edge.
    """
    # Ensure binary
    lesion_mask = (binary_dilation(lesion_mask, dilation_margin) > 0).astype(np.uint8)    
    vessel_mask = (vessel_mask > 0).astype(np.uint8)

    # Perform erosion
    vessel_edge = vessel_mask - binary_erosion(vessel_mask).astype(np.uint8)

    merged = lesion_mask + 2 * vessel_edge

    # Contact (convolution-style edge touching) between lesion and vessel
    tumor_contact = get_contact(merged)

    return merged, tumor_contact, vessel_edge


def get_contact(merged_mask: np.array) -> int:
    """
    Computes contact-based involvement between lesion and vessel in a merged mask.

    Parameters
    ----------
    merged_mask: np.ndarray
        Array containing lesion (=1), vessel edge wo contact (=2), vessel edge in contact (=3).

    Returns
    -------
    tumor_contact: int
        Number of pixels of the vessel that are invaded by the tumor.
    """

    kernels = [
        np.array([[1, 1]]), # horizontal
        np.array([[1], [1]]), # vertical
        np.array([[1, 0], [0, 1]]), # diagonal (top-left to bottom-right)
        np.array([[0, 1], [1, 0]]) # diagonal (top-right to bottom-left)
    ]

    # Create binary masks
    lesion_only = (merged_mask == 1).astype(np.uint8)
    vessel_only = (merged_mask == 2).astype(np.uint8)
    overlap = (merged_mask == 3).astype(np.uint8)

    tumor_contact = np.count_nonzero(overlap)

    for kernel in kernels:
        lesion_dilated = convolve(lesion_only, kernel, mode='constant', cval=0)
        vessel_contact = (lesion_dilated > 0) & (vessel_only == 1)
        tumor_contact += np.count_nonzero(vessel_contact)

    return tumor_contact


def get_plane(plane, scan, slice_id):
    if plane == 'ax':
        return scan[:, :, slice_id]
    elif plane == 'sag':
        return scan[:, slice_id, :]
    elif plane == 'cor':
        return scan[slice_id, :, :]
    

def get_selected_segments(slice, vessel):
    # Slice: contains the segmentations
    # Vessel: integer indicating which vessel
    for j in range(0, 23):
        if j == vessel or j == 20: continue
        # not vessel and not tumor structures
        slice = np.where(slice == j, 0, slice)
    # vessel structure
    slice = np.where(slice == vessel, 1, slice)
    # tumor structure
    slice = np.where(slice == 1, 1, slice)

    return slice


def crop_slices_together(slice_x: np.array, slice_y: np.array, 
                         image: np.array=None, padding=20) -> tuple[np.array, np.array, np.array]:
    """
    Crops both slices using the same bounding box based on non-zero values in either mask.

    Parameters
    ----------
        slice_x: np.ndarray
            First mask
        slice_y: np.ndarray
            Second mask
        image: np.ndarray
            CT image
        padding: int
            Optional padding around the cropped region.

    Returns
    ----------
        slice_x[r1:r2, c1:c2]
            Cropped version of slice_x.
        slice_x[r1:r2, c1:c2]
            Cropped version of slice_x.
        image[r1:r2, c1:c2]
            Cropped version of the CT image.
    """
    
    assert slice_x.shape == slice_y.shape, "Both slices must have the same shape"

    # Combine masks to get a union of non-zero regions
    combined_mask = (slice_x != 0) | (slice_y != 0)
    non_zero_indices = np.argwhere(combined_mask)

    if non_zero_indices.size == 0:
        # Nothing to crop; return original
        return slice_x, slice_y

    # Bounding box coordinates
    top_left = non_zero_indices.min(axis=0) - padding
    bottom_right = non_zero_indices.max(axis=0) + 1 + padding

    # Clamp to image boundaries
    top_left = np.maximum(top_left, 0)
    bottom_right = np.minimum(bottom_right, slice_x.shape)

    r1, c1 = top_left
    r2, c2 = bottom_right

    if image is not None: return slice_x[r1:r2, c1:c2], slice_y[r1:r2, c1:c2], image[r1:r2, c1:c2]
    else: return slice_x[r1:r2, c1:c2], slice_y[r1:r2, c1:c2], None