import numpy as np
import torch
from scipy.stats import norm
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from scipy.integrate import quad
from scipy.interpolate import interp1d
from monai.transforms import CropForeground

from torchmetrics.classification import MulticlassCalibrationError


'''
Prepare the annotations and results to be processed
'''

def preprocess_results(ct_image, annotations, results):
    """
    Preprocess the images, predictions and annotations in order to be evaluated.
    It crops the foreground and applies the same crop to the rest of matrices.
    This is done to save some memory and work with smaller matrices.
    
    ct_image: CT images of shape (slices, X, Y)
    annotations: list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                 each gt has the following shape (slices, X, Y)
    results: list containing the results [binarized prediction, 
                                          pancreas_confidence,
                                          kidney_confidence,
                                          liver_confidence
                                         ]
            the binarized prediction the following values: 1: pancreas, 2: kidney, 3: liver
            each confidence has probabilistic values that range from 0 to 1
     
    @output cropped_annotations, cropped_results[0], cropped_results[1:]
  
    """
    
    # Define the CropForeground transform
    cropper = CropForeground(select_fn=lambda x: x > 0)  # Assuming non-zero voxels are foreground

    # Compute the cropping box based on the CT image
    box_start, box_end = cropper.compute_bounding_box(ct_image)
    
    # Apply the cropping box to all annotations
    cropped_annotations = [annotation[..., box_start[0]:box_end[0], box_start[1]:box_end[1]] for annotation in annotations]
    cropped_results = [result[..., box_start[0]:box_end[0], box_start[1]:box_end[1]] for result in results]

    return cropped_annotations, cropped_results[0], cropped_results[1:]


'''
Dice Score Evaluation
'''

def consensus_dice_score(groundtruth, bin_pred, prob_pred):
    """
    Computes an average of dice score for consensus areas only.
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    bin_pred: binarized prediction matrix containing values: {0,1,2,3}
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
     
    @output dice_scores, confidence
  
    """
    
    # Transform probability predictions to one-hot encoding by taking the argmax
    prediction_onehot = AsDiscrete(to_onehot=4)(torch.from_numpy(np.expand_dims(bin_pred, axis=0)))[1:].astype(np.uint8)
    
    # Split ground truth into separate organs and calculate consensus
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}
    consensus = {}
    dissensus = {}

    for organ_val, organ_name in organs.items():
        # Get the ground truth for the current organ
        organ_gt = (groundtruth == organ_val).astype(np.uint8)
        organ_bck = (groundtruth != organ_val).astype(np.uint8)
        
        # Calculate consensus regions (all annotators agree)
        consensus[organ_name] = np.logical_and.reduce(organ_gt, axis=0).astype(np.uint8)
        consensus[f"{organ_name}_bck"] = np.logical_and.reduce(organ_bck, axis=0).astype(np.uint8)
        
        # Calculate dissensus regions (where both background and foreground are 0)
        dissensus[organ_name] = np.logical_and(consensus[organ_name] == 0, 
                                               consensus[f"{organ_name}_bck"]== 0).astype(np.uint8)
    
    # Mask the predictions and ground truth with the consensus areas
    predictions = {}
    groundtruth_consensus = {}
    confidence = {}

    for organ_val, organ_name in organs.items():
        # Apply the dissensus mask to exclude non-consensus areas
        filtered_prediction = prediction_onehot[organ_val-1] * (1 - dissensus[organ_name])
        filtered_groundtruth = consensus[organ_name] * (1 - dissensus[organ_name])
        
        predictions[organ_name] = filtered_prediction
        groundtruth_consensus[organ_name] = filtered_groundtruth
        
        # Compute mean probabilities and confidence in the consensus area
        prob_in_consensus_organ = prob_pred[organ_val-1] * np.where(consensus[organ_name]==1, 1, np.nan)
        prob_in_consensus_bck = prob_pred[organ_val-1] * np.where(consensus[f"{organ_name}_bck"]==1, 1, np.nan)
        mean_conf_organ = np.nanmean(prob_in_consensus_organ)
        mean_conf_bck = np.nanmean(prob_in_consensus_bck)        
        confidence[organ_name] = (((1-mean_conf_bck)+mean_conf_organ)/2)
    
    # Create DiceMetric instance
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)

    dice_scores = {}
    for organ_name in organs.values():
        gt = torch.from_numpy(groundtruth_consensus[organ_name])
        pred = torch.from_numpy(predictions[organ_name])
        dice_metric.reset()
        dice_metric(pred, gt)
        dice_scores[organ_name] = dice_metric.aggregate().item()
    
    return dice_scores, confidence
    

'''
Volume Assessment
'''

def volume_metric(groundtruth, prediction, voxel_proportion=1):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
    voxel_proportion: vaue of the resampling needed voxel-wise, 1 by default
     
    @output crps_dict
    """
    
    cdf_list = calculate_volumes_distributions(groundtruth, voxel_proportion)
        
    crps_dict = {}    
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}

    for organ_val, organ_name in organs.items():
        probabilistic_volume = compute_probabilistic_volume(prediction[organ_val-1], voxel_proportion)
        crps_dict[organ_name] = crps_computation(probabilistic_volume, cdf_list[organ_name], mean_gauss[organ_name], var_gauss[organ_name])

    return crps_dict


def heaviside(x):
    return 0.5 * (np.sign(x) + 1)


def crps_computation(predicted_volume, cdf, mean, std_dev):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    predicted_volume: scalar value representing the volume obtained from the 
                        probabilistic prediction
    cdf: cumulative density distribution (CDF) of the groundtruth volumes
    mean: mean of the gaussian distribution obtained from the three groundtruth volumes
    std_dev: std_dev of the gaussian distribution obtained from the three groundtruth volumes
     
    @output crps_dict
    """
    
    def integrand(y):
        return (cdf(y) - heaviside(y - predicted_volume)) ** 2
    
    lower_limit = mean - 3 * std_dev
    upper_limit = mean + 3 * std_dev
    
    crps_value, _ = quad(integrand, lower_limit, upper_limit)
        
    return crps_value


def calculate_volumes_distributions(groundtruth, voxel_proportion=1):
    """
    Calculates the Cumulative Distribution Function (CDF) of the Probabilistic Function Distribution (PDF)
    obtained by calcuating the mean and the variance of considering the three annotations.
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    voxel_proportion: vaue of the resampling needed voxel-wise, 1 by default
    
    @output cdfs_dict
    """
    
    organs = {1: 'panc', 2: 'kidn', 3: 'livr'}
    
    global mean_gauss, var_gauss, volumes  # Make them global to access in crps
    mean_gauss = {}
    var_gauss = {}
    volumes = {}

    for organ_val, organ_name in organs.items():
        volumes[organ_name] = [np.unique(gt, return_counts=True)[1][organ_val] * np.prod(voxel_proportion) for gt in groundtruth]
        mean_gauss[organ_name] = np.mean(volumes[organ_name])
        var_gauss[organ_name] = np.std(volumes[organ_name])

    # Create normal distribution objects
    gaussian_dists = {organ_name: norm(loc=mean_gauss[organ_name], scale=var_gauss[organ_name]) for organ_name in organs.values()}
    
    # Generate CDFs
    cdfs = {}
    for organ_name in organs.values():
        x = np.linspace(gaussian_dists[organ_name].ppf(0.01), gaussian_dists[organ_name].ppf(0.99), 100)
        cdf_values = gaussian_dists[organ_name].cdf(x)
        cdfs[organ_name] = interp1d(x, cdf_values, bounds_error=False, fill_value=(0, 1))  # Create an interpolation function

    return cdfs
    
    
def compute_probabilistic_volume(preds, voxel_proportion=1):
    """
    Computes the volume of the matrix given (either pancreas, kidney or liver)
    by adding up all the probabilities in this matrix. This way the uncertainty plays
    a role in the computation of the predicted organ. If there is no uncertainty, the 
    volume should be close to the mean obtained by averaging the three annotations.
    
    preds: probabilistic matrix of a specific organ
    voxel_proportion: vaue of the resampling needed voxel-wise, 1 by default
     
    @output volume
    """
    
    # Sum the predicted probabilities to get the volume
    volume = preds.sum().item()
    
    return volume*voxel_proportion


'''
Expected Calibration Error
'''

def multirater_expected_calibration_error(annotations_list, prob_pred):
    """
    Returns a list of length three of the Expected Calibration Error (ECE) per annotation.
    
    annotations_list: list of length three containing the three annotations
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
     
    @output ece_dict
    """
    
    ece_dict = {}

    for e in range(3):
        ece_dict[e] = expected_calibration_error(annotations_list[e], prob_pred)
        
    return ece_dict


def expected_calibration_error(groundtruth, prob_pred_onehot, num_classes=4, n_bins=50):
    """
    Computes the Expected Calibration Error (ECE) between the given annotation and the 
    probabilistic prediction
    
    groundtruth: groundtruth matrix containing the following values: 1: pancreas, 2: kidney, 3: liver
                    shape: (slices, X, Y)
    prob_pred_onehot: probability prediction matrix, shape: (3, slices, X, Y), the three being
                    a probability matrix per each class
    num_classes: number of classes
    n_bins: number of bins                    
                    
    @output ece
    """ 
    
    # Convert inputs to torch tensors
    all_groundtruth = torch.tensor(groundtruth)
    all_samples = torch.tensor(prob_pred_onehot)
    
    # Calculate the probability for the background class
    background_prob = 1 - all_samples.sum(dim=0, keepdim=True)
    
    # Combine background probabilities with the provided probabilities
    all_samples_with_bg = torch.cat((background_prob, all_samples), dim=0)
    
    # Flatten the tensors to (num_samples, num_classes) and (num_samples,)
    all_groundtruth_flat = all_groundtruth.view(-1)
    all_samples_flat = all_samples_with_bg.permute(1, 2, 3, 0).reshape(-1, num_classes)
    
    # Initialize the calibration error metric
    calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins)

    # Calculate the ECE
    ece = calibration_error(all_samples_flat, all_groundtruth_flat).cpu().detach().numpy().astype(np.float64)
    
    return ece
