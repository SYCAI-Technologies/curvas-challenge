from pathlib import Path

from glob import glob
import SimpleITK as sitk
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
    

def run():
        
    # Read the input
    input_thoracic_abdominal_ct_image, input_metadata = load_image_file_as_array(
        location=INPUT_PATH / "images/thoracic-abdominal-ct",
    )
        
    # Make the predictions
    output_abdominal_organ_segmentation, output_pancreas_confidence, output_kidney_confidence, output_liver_confidence = perform_inference(
        input_image=input_thoracic_abdominal_ct_image, input_metadata=input_metadata
    )

    # Save the output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/abdominal-organ-segmentation",
        array=output_abdominal_organ_segmentation,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/kidney-confidence",
        array=output_kidney_confidence,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/pancreas-confidence",
        array=output_pancreas_confidence,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/liver-confidence",
        array=output_liver_confidence,
    )
    
    return 0


def perform_inference(input_image, input_metadata):
    # Save the input image temporarily to use with nnU-Net
    temp_output_path = OUTPUT_PATH / "temp_output"
    temp_output_path.mkdir(parents=True, exist_ok=True)
    temp_input_path = OUTPUT_PATH / "temp_input_image.nii.gz"

    sitk.WriteImage(sitk.GetImageFromArray(input_image), str(temp_input_path))
    
    # Define nnUNet v2 model parameters
    model_name = '3d_fullres'
    trainer_class_name = 'nnUNetTrainer'
    plans_identifier = 'nnUNetPlans'
    task_name = 'Dataset475_CURVAS'
    
    # Create nnUNet predictor        
    predictor = nnUNetPredictor(
            tile_step_size=1,
            use_gaussian=True,
            use_mirroring=False,
            device=torch.device('cuda'),
            perform_everything_on_device=True,
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=False
        )
    
    predictor.initialize_from_trained_model_folder(
            "/opt/algorithm/nnUNet_results/{}/{}__{}__{}".format(task_name, 
                                                                trainer_class_name,
                                                                plans_identifier,
                                                                model_name),
            use_folds=("0"),
            checkpoint_name="checkpoint_best.pth",
    )

    # Perform prediction
    predictor.predict_from_files(
                    list_of_lists_or_source_folder=[[str(temp_input_path)]],
                    output_folder_or_list_of_truncated_output_files=[str(temp_output_path)],
                    overwrite=True,
                    save_probabilities=True,
                    num_processes_preprocessing=6, 
                    num_processes_segmentation_export=6,
                    num_parts=6#, part_id=0
                )
    
    # Load the nnUNet output
    output_file_path = OUTPUT_PATH / "temp_output.nii.gz"
    output_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(str(output_file_path)))
    
    output_file_path_probs = OUTPUT_PATH / "temp_output.npz"
    probabilities = np.load(output_file_path_probs, allow_pickle=False)['probabilities'] # (4, 512, 512, slices)

    return output_segmentation.astype(np.uint8), probabilities[1],  probabilities[2], probabilities[3]


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = sitk.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return sitk.GetArrayFromImage(result), result.GetSpacing()


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tiff to match the expected output
    suffix = ".mha"

    image = sitk.GetImageFromArray(array)
    sitk.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run())
