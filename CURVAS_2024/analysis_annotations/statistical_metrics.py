import numpy as np

from scipy.spatial.distance import directed_hausdorff
from statsmodels.stats.inter_rater import fleiss_kappa
from pingouin import intraclass_corr
import pandas as pd



# Function to compute DSC for binary segmentations
def dice_similarity(seg1, seg2):
    intersection = np.sum((seg1 > 0) & (seg2 > 0))
    volume_sum = np.sum(seg1 > 0) + np.sum(seg2 > 0)
    if volume_sum == 0:
        return 1.0  # Both segmentations are empty
    return 2.0 * intersection / volume_sum



# Function to isolate structure voxels (non-background)
def isolate_structure_voxels(seg1, seg2):
    mask = (seg1 > 0) | (seg2 > 0)  # Assuming 0 is the background label
    return seg1[mask], seg2[mask]


def isolate_structure_voxels_list(segmentations):
    """
    Isolate the structure voxels where at least one of the segmentations has a structure (non-background).
    Apply this mask across all segmentations.
    
    Args:
    segmentations (list of numpy arrays): List of segmentations (from different raters).
    
    Returns:
    list of numpy arrays: Flattened arrays of voxels corresponding to structure only for all segmentations.
    """
    # Create a mask of voxels where at least one segmentation is non-zero (structure)
    structure_voxel_mask = np.any(np.stack(segmentations) > 0, axis=0)
    
    # Apply the mask to each segmentation to only keep the voxels of interest (structure)
    return [seg[structure_voxel_mask].flatten() for seg in segmentations]



# Function to compute Hausdorff Distance
def hausdorff_distance(seg1, seg2):
    # Find coordinates of non-background voxels (structure points)
    points1 = np.argwhere(seg1 > 0)  # Non-background points in seg1
    points2 = np.argwhere(seg2 > 0)  # Non-background points in seg2
    
    # Compute directed Hausdorff distances in both directions
    d_forward = directed_hausdorff(points1, points2)[0]
    d_backward = directed_hausdorff(points2, points1)[0]
    
    # Symmetric Hausdorff Distance (maximum of both directions)
    return max(d_forward, d_backward)



def compute_fleiss_kappa(segmentations_flat):
    # Prepare input as a list of counts of each label (0 for background, 1 for structure) for each voxel
    voxel_labels = np.array(segmentations_flat).T  # Transpose to align raters
    count_data = []
    
    # Convert to Fleiss format (count of 0's and 1's for each voxel across raters)
    for voxel_label in voxel_labels:
        counts = np.bincount(voxel_label, minlength=2)
        count_data.append(counts)
    
    return fleiss_kappa(np.array(count_data))



def compute_icc(segmentations_flat):
    # Prepare data for ICC calculation
    data = []
    for rater, segmentation in enumerate(segmentations_flat):
        for voxel_idx, value in enumerate(segmentation):
            data.append([voxel_idx, rater, value])
    
    df = pd.DataFrame(data, columns=['voxel', 'rater', 'rating'])
    icc_result = intraclass_corr(data=df, targets='voxel', raters='rater', ratings='rating')
    
    # Get the ICC for consistency or agreement
    return icc_result['ICC'].iloc[0]  # Return the first ICC result


from joblib import Parallel, delayed

def parallel_compute_icc(segmentations):
    """
    Compute ICC in parallel for flattened segmentations (already 1D).
    Assumes `segmentations` is a list of 1D flattened arrays for different raters.
    """
    # Since the segmentations are already flattened, no need to slice over dimensions
    def compute_icc_for_flat():
        return compute_icc(segmentations)

    # Run the computation in parallel without needing to iterate over slices
    icc_values = Parallel(n_jobs=20)(
        delayed(compute_icc_for_flat)() for _ in range(len(segmentations))
    )

    # Return the mean ICC value across all segmentations
    return np.mean(icc_values)


def parallel_compute_fleiss_kappa(flattened_segmentations):
    """
    Compute Fleiss Kappa for each slice of flattened segmentations in parallel.

    Args:
    flattened_segmentations (list of numpy arrays): List of flattened segmentations for multiple raters.

    Returns:
    float: Mean Fleiss Kappa score across all slices.
    """
    # Assuming all flattened_segmentations are of the same length
    n_raters = len(flattened_segmentations)

    # Prepare counts for Fleiss Kappa calculation
    def prepare_fleiss_data():
        counts = []

        for i in range(len(flattened_segmentations[0])):  # Loop over each voxel
            # Create a count for each category
            count = [0, 0, 0]  # Assuming 3 categories: background (0), structure1 (1), structure2 (2)
            for rater_idx in range(n_raters):
                category = flattened_segmentations[rater_idx][i]
                count[category] += 1
            counts.append(count)
        
        return counts

    # Compute Fleiss Kappa
    counts = prepare_fleiss_data()
    fleiss_value = fleiss_kappa(counts)

    return fleiss_value