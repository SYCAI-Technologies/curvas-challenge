import numpy as np
import torch
from scipy.stats import norm
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from scipy.integrate import quad
from scipy.interpolate import interp1d


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
    prediction_onehot = AsDiscrete(to_onehot=4)(np.expand_dims(bin_pred, axis=0))[1:]
    
    # Split ground truth into separate organs and calculate consensus
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}
    consensus = {}
    dissensus = {}

    for organ_val, organ_name in organs.items():
        # Get the ground truth for the current organ
        organ_gt = (groundtruth == organ_val).astype(int)
        organ_bck = (groundtruth != organ_val).astype(int)
        
        # Calculate consensus regions (all annotators agree)
        consensus[organ_name] = np.logical_and.reduce(organ_gt, axis=0).astype(int)
        consensus[f"{organ_name}_bck"] = np.logical_and.reduce(organ_bck, axis=0).astype(int)
        
        # Calculate dissensus regions (where both background and foreground are 0)
        dissensus[organ_name] = np.logical_and(consensus[organ_name] == 0, consensus[f"{organ_name}_bck"] == 0).astype(int)
    
    # Mask the predictions and ground truth with the consensus areas
    predictions = {}
    groundtruth_consensus = {}
    mean_probs = {}
    confidence = {}

    for organ_val, organ_name in organs.items():    
        # Apply the dissensus mask to exclude non-consensus areas
        filtered_prediction = prediction_onehot[organ_val-1] * (1 - dissensus[organ_name])
        filtered_groundtruth = consensus[organ_name] * (1 - dissensus[organ_name])
        
        predictions[organ_name] = filtered_prediction
        groundtruth_consensus[organ_name] = filtered_groundtruth
        
        # Compute mean probabilities and confidence in the consensus area
        prob_in_consensus = prob_pred[organ_val-1] * consensus[organ_name]
        confidence[organ_name] = np.max(prob_in_consensus[consensus[organ_name] == 1])
    
    # Create DiceMetric instance
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)

    dice_scores = {}
    for organ_name in organs.values():
        gt = torch.from_numpy(groundtruth_consensus[organ_name])#[None, ...]
        pred = torch.from_numpy(predictions[organ_name])#[None, ...]
        dice_metric.reset()
        dice_metric(pred, gt)
        dice_scores[organ_name] = dice_metric.aggregate().item()
    
    return dice_scores, confidence


'''
Volume Assessment
'''

def volume_metric(groundtruth, prediction, pixdim=1):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
    pixdim: vaue of the resampling needed, 1 by default
     
    @output crps_dict
    """
    
    cdf_list = calculate_volumes_distributions(groundtruth, pixdim)
        
    crps_dict = {}    
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}

    for organ_val, organ_name in organs.items():
        probabilistic_volume = compute_probabilistic_volume(prediction[organ_val-1])
        crps_dict[organ_name] = crps_computation(probabilistic_volume, cdf_list[organ_name], mean_gauss[organ_name], var_gauss[organ_name])

    return crps_dict


def heaviside(x):
    return 0.5 * (np.sign(x) + 1)


def crps_computation(predicted_volume, cdf, mean, std_dev, logging=False):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
    pixdim: vaue of the resampling needed, 1 by default
     
    @output crps_dict
    """
    
    def integrand(y):
        return (cdf(y) - heaviside(y - predicted_volume)) ** 2
    
    lower_limit = mean - 3 * std_dev
    upper_limit = mean + 3 * std_dev
    
    crps_value, _ = quad(integrand, lower_limit, upper_limit)
    
    if logging:
        # Debug information
        print(f"Predicted Volume: {predicted_volume}")
        print(f"CRPS Value: {crps_value}")
        print(f"CDF at predicted volume: {cdf(predicted_volume)}")
        print(f"Integral range: {lower_limit} to {upper_limit}")
    
    return crps_value


def calculate_volumes_distributions(groundtruth, pixdim=1, logging=False):
    """
    Calculates the Cumulative Distribution Function (CDF) of the Probabilistic Function Distribution (PDF)
    obtained by calcuating the mean and the variance of considering the three annotations.
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    pixdim: vaue of the resampling needed, 1 by default            
    
    @output cdfs_dict
    """
    
    organs = {1: 'panc', 2: 'kidn', 3: 'livr'}
    
    global mean_gauss, var_gauss, volumes  # Make them global to access in crps
    mean_gauss = {}
    var_gauss = {}
    volumes = {}

    for organ_val, organ_name in organs.items():
        volumes[organ_name] = [np.unique(gt, return_counts=True)[1][organ_val] * np.prod(pixdim) for gt in groundtruth]
        mean_gauss[organ_name] = np.mean(volumes[organ_name])
        var_gauss[organ_name] = np.std(volumes[organ_name])

    if logging:
        # Debug information
        print(f"volumes: {volumes}")
        print(f"mean_gauss: {mean_gauss}")
        print(f"var_gauss: {var_gauss}")

    # Create normal distribution objects
    gaussian_dists = {organ_name: norm(loc=mean_gauss[organ_name], scale=var_gauss[organ_name]) for organ_name in organs.values()}
    
    # Generate CDFs
    cdfs_dict = {}
    for organ_name in organs.values():
        x = np.linspace(gaussian_dists[organ_name].ppf(0.01), gaussian_dists[organ_name].ppf(0.99), 100)
        cdf_values = gaussian_dists[organ_name].cdf(x)
        cdfs_dict[organ_name] = interp1d(x, cdf_values, bounds_error=False, fill_value=(0, 1))  # Create an interpolation function

    return cdfs_dict
    
    
def compute_probabilistic_volume(preds):
    """
    Computes the volume of the matrix given (either pancreas, kidney or liver)
    by adding up all the probabilities in this matrix. This way the uncertainty plays
    a role in the computation of the predicted organ. If there is no uncertainty, the 
    volume should be close to the mean obtained by averaging the three annotations.
    
    preds: probabilistic matrix of a specific organ
     
    @output volume
    """
    
    # Sum the predicted probabilities to get the volume
    volume = preds.sum().item()
    
    return volume


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
    
    
def expected_calibration_error(groundtruth, prob_pred, M=5):
    """
    Computes the Expected Calibration Error (ECE) between the given annotation and the 
    probabilistic prediction.ñ
    
    groundtruth: groundtruth matrix containing the following values: 1: pancreas, 2: kidney, 3: liver
                    shape: (slices, X, Y)
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
     
    @output ece
    """ 
    
    all_groundtruth = torch.from_numpy(groundtruth)
    all_samples = torch.from_numpy(prob_pred)
    
    # uniform binning approach with M number of bins
    bin_boundaries = torch.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i (confidences) and the final predictions from these confidences
    confidences, predicted_label = torch.max(all_samples, 0)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label.eq(all_groundtruth)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].float().mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece