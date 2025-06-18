import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation


def get_involvement(tumor, vessel, vessel_value, plane):

    max_degrees = {'cor': 0, 'sag': 0, 'ax': 0}
    planes = {'cor': 0, 'sag': 1, 'ax': 2}

    for plane in range(len(planes)):
        merged_masks = np.where(vessel==1, 2, tumor)
        slice_list = np.unique(sorted(np.where(merged_masks)[plane]))

        for slice_id in slice_list:
            slice_tumor = get_plane(planes[plane], tumor, slice_id)
            if 1 not in slice_tumor: continue

            slice_vessels = get_plane(planes[plane], vessel, slice_id)
            slice_vessel = get_selected_segments(slice_vessels, vessel_value)
            if 1 not in slice_vessel: continue

            cropped_slice_tumor, cropped_slice_vessel = crop_slices_together(slice_tumor, slice_vessel)
            slice_tumor_dilated = binary_dilation(cropped_slice_tumor)
            _, _, tumor_contact, vessel_edge = assess_vascular_involvement(slice_tumor_dilated, cropped_slice_vessel, dilation_margin=3)

            degrees = (tumor_contact / np.sum(vessel_edge == 1) * 360) if np.sum(vessel_edge == 1) > 0 else 0
            
            if max_degrees[planes[plane]] < degrees: max_degrees[planes[plane]] = degrees

    return max_degrees


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


def crop_slices_together(slice_x, slice_y, image, padding=1):
    """
    Crops both slices using the same bounding box based on non-zero values in either mask.

    Args:
        slice_x (np.ndarray): First mask (e.g., lesion).
        slice_y (np.ndarray): Second mask (e.g., vessel).
        padding (int): Optional padding around the cropped region.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped versions of slice_x and slice_y.
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

    return slice_x[r1:r2, c1:c2], slice_y[r1:r2, c1:c2]
  
    
def assess_vascular_involvement(lesion_mask, vessel_mask, dilation_margin=1, distance_threshold=3):
    """
    Assess direct overlap and proximity-based vessel involvement.

    Args:
        lesion_mask (np.ndarray): Binary mask with 1 = lesion.
        vessel_mask (np.ndarray): Binary mask with 1 = vessel.
        dilation_margin (int): Number of pixels to dilate lesion mask.

    Returns:
        dict: overlap ratio, proximity ratio, direct contact ratio.
    """
    # Ensure binary
    lesion_mask = (lesion_mask > 0).astype(np.uint8)    
    vessel_mask = (vessel_mask > 0).astype(np.uint8)

    # Perform erosion
    vessel_edge = vessel_mask - binary_erosion(vessel_mask).astype(np.uint8)

    merged = lesion_mask + 2 * vessel_edge

    # Contact (convolution-style edge touching) between lesion and vessel
    contact_ratio, tumor_contact = get_contact_degrees(merged)

    return contact_ratio, merged, tumor_contact, vessel_edge