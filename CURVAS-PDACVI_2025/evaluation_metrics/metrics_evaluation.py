import numpy as np
import torch
from scipy.stats import norm
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import wasserstein_distance
from .vi_metrics import get_involvement

from torchmetrics.classification import MulticlassCalibrationError

'''
Vascular Involvement
'''
vascular_strucutres_dict = {'PORTA': 1, 'SMV': 2, 'AORTA': 3, 'CELIAC TRUNK': 4, 'SMA': 5}
planes = ['cor', 'sag', 'ax']

 
def aggregated_vascular_involvement(gt_distr: dict, pred_distr: dict) -> dict:
    """
    Compute the aggregated values of the Vascular Involvement (VI) for each structure

    Parameters
    ----------
    gt_distr: dict
        Dictionary of distributions.
    pred_distr : dict
        Dictionary of distributions.

    Returns
    -------
    aggregated_metrics: dictionary
        Dictionary of Wasserstein metrics for each vascular structure.
    """

    results = {}
    aggregated_metrics = {structure: 0 for structure in vascular_strucutres_dict}

    for vessel in vascular_strucutres_dict:
        print(vessel)
        results[vessel] = {}
        for plane in ['cor', 'sag', 'ax']:
            gt = gt_distr[vessel][plane]
            pred = pred_distr[vessel][plane]
            print(f'{plane}: gt {gt}, pred {pred}')
            
            x = np.linspace(0, 360, 1000)
            p = gt.pdf(x)
            q = pred.pdf(x)

            # === Fallback rules first ===
            if np.allclose(p, 0.0) and np.allclose(q, 0.0):
                results[vessel][plane] = 0.0
            elif np.allclose(p, 0.0):
                results[vessel][plane] = 1.0
            elif np.allclose(q, 0.0):
                results[vessel][plane] = 1.0
            # === Else, sanitize and compute Wasserstein ===
            else:
                p = np.nan_to_num(p, nan=0.0, neginf=0.0, posinf=0.0)
                q = np.nan_to_num(q, nan=0.0, neginf=0.0, posinf=0.0)
                p = np.clip(p, 0, None)
                q = np.clip(q, 0, None)
                if p.sum() > 0: p /= p.sum()
                if q.sum() > 0: q /= q.sum()
                results[vessel][plane] = wasserstein_distance(x, x, p, q)

        # Aggregate per metric
        aggregated_metrics[vessel] = float(np.mean([results[vessel][plane] for plane in results[vessel]]))
    
    return aggregated_metrics


def vascular_involvement(annotations: list, vi: np.array, soft_mask:np.array, smooth: float = 1e-6):
    """
    Compute the Vascular Involvement (VI) of each structure for both annotations 
    and the probabilistic output. 

    Parameters
    ----------
    pred : np.ndarray
        The soft prediction, values in [0, 1], shape (H, W, D) or (N, H, W, D).
    annotations : np.ndarray
        List of Multiple binary annotations of length K.
        Shape of each element in the list (H, W, D) or (N, H, W, D).
        K is the number of annotators.
    smooth : float
        Smoothing constant to avoid division by zero.

    Returns
    -------
    gt_distribution : dictionary
        Dictionary of Gaussian Distribtuions for the different vascular structures.
    pred_distribution : dictionary
        Dictionary of Gaussian Distribtuions for the different vascular structures.
    """

    mx_dgrs_GT = {'PORTA': [], 'SMV': [], 'AORTA': [], 'CELIAC TRUNK': [], 'SMA': []}
    mx_dgrs_pred = {'PORTA': [], 'SMV': [], 'AORTA': [], 'CELIAC TRUNK': [], 'SMA': []}

    thresholds = np.linspace(0.1, 0.8, 6)

    for vasc_struct in vascular_strucutres_dict.keys():
        print(f'--------- {vasc_struct} : {vascular_strucutres_dict[vasc_struct]} ---------')

        print('vascular involvement for GTs')
        for i in range(5):
            mx_dgrs_GT[vasc_struct].append(get_involvement(annotations[i], vi, vascular_strucutres_dict[vasc_struct]))

        print('vascular involvement for prediction')
        for t in thresholds:
            mx_dgrs_pred[vasc_struct].append(get_involvement(soft_mask > t, vi, vascular_strucutres_dict[vasc_struct]))

    gt_distribution = {
        structure: {'cor': 0, 'sag': 0, 'ax': 0}
        for structure in vascular_strucutres_dict
    }

    pred_distribution = {
        structure: {'cor': 0, 'sag': 0, 'ax': 0}
        for structure in vascular_strucutres_dict
    }

    for vasc_struct in vascular_strucutres_dict.keys():
        print(f'--------- {vasc_struct} : {vascular_strucutres_dict[vasc_struct]} ---------')

        for plane in planes:
            plane_values = [d[plane] for d in mx_dgrs_GT[vasc_struct]]
            pred_mean = np.mean(plane_values)
            pred_std = np.std(plane_values)

            # Define GT distribution
            gt_distribution[vasc_struct][plane] = norm(loc=pred_mean, scale=pred_std + smooth)

            plane_values = [d[plane] for d in mx_dgrs_pred[vasc_struct]]
            pred_mean = np.mean(plane_values)
            pred_std = np.std(plane_values)

            # Define pred distribution
            pred_distribution[vasc_struct][plane] = norm(loc=pred_mean, scale=pred_std + smooth)

    return gt_distribution, pred_distribution


'''
Dice Score Evaluation
'''

def merged_dice_score(pred: np.ndarray, bin_pred: np.ndarray, annotations: list, staple_gt: np.array, smooth: float = 1e-6) -> tuple[float, float]:
    """
    Compute the soft Dice score between a soft prediction and the average of multiple binary annotations.

    Parameters
    ----------
    pred : np.ndarray
        The soft prediction, values in [0, 1], shape (H, W, D) or (N, H, W, D).
    pred_bin: np.array
        Binarized prediction.
    annotations : np.ndarray
        Multiple binary annotations, shape (K, H, W, D) or (K, N, H, W, D).
        K is the number of annotators.
    staple_gt: np.array
        Binary STAPLE GT generate from all 5 annotations.
    smooth : float
        Smoothing constant to avoid division by zero.

    Returns
    -------
    dice : float
        The soft Dice score from soft labels
    bin_dice : float
        The Dice score from hard labels.
    """

    # Average over annotations
    avg_gt = np.mean(np.stack(annotations, axis=0), axis=0)

    # Flatten for Dice computation
    pred_flat = pred.reshape(-1)
    gt_flat = avg_gt.reshape(-1)

    def dice_coefficient(pred, gt):
        intersection = np.sum(pred * gt)
        return 2.0 * intersection / (np.sum(pred) + np.sum(gt) + smooth)
    
    thresholds = np.linspace(0.1, 0.8, 6)

    list_dice = []

    for t in thresholds:
        pred_flat_bin = pred_flat > t
        gt_flat_bin = gt_flat > t

        if np.all(pred_flat_bin == 0) and np.all(gt_flat_bin == 0): list_dice.append(1)
        else:     

            list_dice.append(dice_coefficient(pred_flat_bin, gt_flat_bin))

    ### binary dice
    bin_dice = dice_coefficient(bin_pred, staple_gt)

    return float(np.mean(list_dice)), float(bin_dice)
    

'''
Volume Assessment
'''

def volume_metric(annotations: list, prediction: np.ndarray, voxel_proportion: float=1) -> float:
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    Parameters
    ----------
    annotations: list
        List of Multiple binary annotations of length K.
        Shape of each element in the list (H, W, D) or (N, H, W, D).
        K is the number of annotators.
    prob_pred: np.ndarray
        The soft prediction, values in [0, 1], shape (H, W, D) or (N, H, W, D).
    voxel_proportion: float
        Vaue of the resampling needed voxel-wise, 1 by default
    
    Returns
    -------
    crps float
        CRPS value obtained for the specific prediction.
    """
    
    print('distr')
    cdf = calculate_volumes_distributions(np.stack(annotations), voxel_proportion)
    print('vol')
    probabilistic_volume = compute_probabilistic_volume(prediction, voxel_proportion)
    print('crps')
    crps = crps_computation(probabilistic_volume, cdf, mean_gauss, var_gauss)

    return float(crps)


def heaviside(x):
    return 0.5 * (np.sign(x) + 1)


def crps_computation(predicted_volume: np.array, cdf: interp1d, mean: float, std_dev: float) -> float:
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    Parameters
    ----------
    predicted_volume: float
        scalar value representing the volume obtained from the 
                        probabilistic prediction
    cdf: cdf
        Cumulative density distribution (CDF) of the groundtruth volumes
    mean: float
        Mean of the Gaussian Distribution obtained from the three groundtruth volumes
    std_dev: float
        Standard Deviation of the Gaussian Distribution obtained from the groundtruth volumes
    
    Returns
    -------
    crps float
        CRPS value obtained for the specific prediction.
    """
    
    def integrand(y):
        return (cdf(y) - heaviside(y - predicted_volume)) ** 2
    
    lower_limit = mean - 3 * std_dev
    upper_limit = mean + 3 * std_dev
    
    crps_value, _ = quad(integrand, lower_limit, upper_limit)
        
    return crps_value


def calculate_volumes_distributions(groundtruth: list, voxel_proportion: float=1) -> interp1d:
    """
    Calculates the Cumulative Distribution Function (CDF) of the Probabilistic Function Distribution (PDF)
    obtained by calcuating the mean and the variance of considering the three annotations.
    
    groundtruth: list
        List of Multiple binary annotations of length K.
        Shape of each element in the list (H, W, D) or (N, H, W, D).
        K is the number of annotators.
    voxel_proportion: float
        vaue of the resampling needed voxel-wise, 1 by default
    
    Returns
    -------
    cdf: interp1d
        Cumulative Distribution function
    """
    
    global mean_gauss, var_gauss, volumes  # Make them global to access in crps

    volumes = [np.sum(gt==1) * np.prod(voxel_proportion) for gt in groundtruth]
    mean_gauss = np.mean(volumes)
    var_gauss = np.std(volumes)

    # Create normal distribution objects
    gaussian_dists = norm(loc=mean_gauss, scale=var_gauss)
    
    # Generate CDFs
    x = np.linspace(gaussian_dists.ppf(0.01), gaussian_dists.ppf(0.99), 100)
    cdf_values = gaussian_dists.cdf(x)
    cdf = interp1d(x, cdf_values, bounds_error=False, fill_value=(0, 1))  # Create an interpolation function

    return cdf
    
    
def compute_probabilistic_volume(preds: np.array, voxel_proportion: float=1) -> float:
    """
    Computes the volume of the matrix given (either pancreas, kidney or liver)
    by adding up all the probabilities in this matrix. This way the uncertainty plays
    a role in the computation of the predicted organ. If there is no uncertainty, the 
    volume should be close to the mean obtained by averaging the three annotations.
    
    Parameters
    ----------
    preds: np.array
        probabilistic matrix of the PDAC
    voxel_proportion: float
        vaue of the resampling needed voxel-wise, 1 by default
     
    Returns
    -------
    volume: float
        volume in mm3 of the PDAC
    """
    
    # Sum the predicted probabilities to get the volume
    volume = preds.sum().item()
    
    return volume*voxel_proportion


'''
Expected Calibration Error
'''

def multirater_expected_calibration_error(annotations_list: list, prob_pred: np.array) -> dict:
    """
    Returns a list of length three of the Expected Calibration Error (ECE) per annotation.
    
    Parameters
    ----------
    annotations_list: list 
        List of Multiple binary annotations of length K.
        Shape of each element in the list (H, W, D) or (N, H, W, D).
        K is the number of annotators.
    prob_pred: np.array
        probabilistic matrix of the PDAC
     
    Returns
    -------
    ece_dict: dict
        Dictionary of the different volumes per annotator
    """
    
    ece_dict = {}

    bbox = get_combined_bbox(annotations_list)

    print(bbox)
    print(annotations_list[0][bbox].shape)
    print(prob_pred[bbox].shape)

    for e in range(5):
        ece_dict[e] = expected_calibration_error(annotations_list[e][bbox], prob_pred[bbox])
        
    return ece_dict


def expected_calibration_error(groundtruth: np.array, prob_pred_onehot: np.array, num_classes: int=2, n_bins: int=50) -> float:
    """
    Computes the Expected Calibration Error (ECE) between the given annotation and the 
    probabilistic prediction
    
    Parameters
    ----------
    groundtruth: np.array
        groundtruth matrix
    prob_pred_onehot: np.array
        probability prediction matrix
    num_classes: int
        number of classes
    n_bins: int
        number of bins                    
                    
    Returns
    -------
    ece: float
        ECE value computed
    """
    
    # Convert inputs to torch tensors
    all_groundtruth = torch.tensor(groundtruth)
    all_samples = torch.tensor(prob_pred_onehot)

    # If binary segmentation and missing class dimension, unsqueeze to [1, D, H, W]
    if all_samples.ndim == 3:
        all_samples = all_samples.unsqueeze(0)  # Make shape [1, D, H, W]
    
    # Handle binary case: only foreground probs are provided
    if all_samples.shape[0] == 1 and num_classes == 2:
        # Add background = 1 - foreground
        background_prob = 1 - all_samples
        all_samples_with_bg = torch.cat((background_prob, all_samples), dim=0)  # Shape: (2, slices, X, Y)
    elif all_samples.shape[0] == num_classes - 1 and num_classes > 2:
        # If you’re missing background, compute it
        background_prob = 1 - all_samples.sum(dim=0, keepdim=True)
        all_samples_with_bg = torch.cat((background_prob, all_samples), dim=0)  # Shape: (num_classes, slices, X, Y)
    elif all_samples.shape[0] == num_classes:
        # Already complete
        all_samples_with_bg = all_samples
    else:
        raise ValueError(f"Shape mismatch: prob_pred_onehot has shape {all_samples.shape}, but num_classes is {num_classes}")

    # Normalize predictions to avoid zero-sum errors
    all_samples_with_bg = torch.clamp(all_samples_with_bg, min=1e-8)  # avoid zero
    all_samples_with_bg = all_samples_with_bg / all_samples_with_bg.sum(dim=0, keepdim=True)
    
    # Flatten the tensors to (num_samples, num_classes) and (num_samples,)
    all_groundtruth_flat = all_groundtruth.reshape(-1)
    all_samples_flat = all_samples_with_bg.permute(1, 2, 3, 0).reshape(-1, num_classes)
    
    # Initialize the calibration error metric
    calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins)

    # Calculate the ECE
    ece = float(calibration_error(all_samples_flat, all_groundtruth_flat).cpu().detach().numpy())
    
    return ece


def get_combined_bbox(volume_list: list) -> tuple:
    """
    Given a list of 3D NumPy arrays, returns a bounding box that includes
    all non-zero voxels from all volumes.

    Parameters
    ----------
    volume_list: list
        List of 3D arrays.

    Returns
    -------
        tuple of slices: (zmin:zmax, ymin:ymax, xmin:xmax)
    """
    # Initialize min and max with extreme values
    zmin, ymin, xmin = np.inf, np.inf, np.inf
    zmax, ymax, xmax = -np.inf, -np.inf, -np.inf

    for vol in volume_list:
        if not np.any(vol):
            continue  # skip empty volumes

        # Get non-zero voxel indices
        coords = np.argwhere(vol > 0)

        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0)

        # Update global bbox
        zmin, ymin, xmin = min(zmin, z0), min(ymin, y0), min(xmin, x0)
        zmax, ymax, xmax = max(zmax, z1), max(ymax, y1), max(xmax, x1)

    # Ensure values are int
    return (slice(int(zmin) - 20, int(zmax) + 20),
            slice(int(ymin) - 20, int(ymax) + 20),
            slice(int(xmin) - 20, int(xmax) + 20))