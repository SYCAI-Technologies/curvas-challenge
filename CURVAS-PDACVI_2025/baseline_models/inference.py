from pathlib import Path

from glob import glob
import SimpleITK
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import numpy as np
import time

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")
import sys
sys.path.append("/opt/app")

import torch
torch_load_original = torch.load

def trusted_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return torch_load_original(*args, **kwargs)

torch.load = trusted_torch_load

def run():
    input_thoracic_abdominal_ct_image, input_metadata = load_image_file_as_array(
        location=INPUT_PATH / "images/thoracic-abdominal-ct",
    )

    print('input_thoracic_abdominal_ct_image: '+str(input_thoracic_abdominal_ct_image.shape))
    print('input_metadata: '+str(input_metadata))
    
    _show_torch_cuda_info()

    start_time = time.time()

    output_pdac_segmentation, output_pdac_confidence = perform_inference(
        input_image=input_thoracic_abdominal_ct_image, input_metadata=input_metadata
    )

    print('output_pdac_segmentation: '+str(output_pdac_segmentation.shape))
    print('output_pdac_confidence: '+str(output_pdac_confidence.shape))

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/pdac-segmentation",
        array=output_pdac_segmentation,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/pdac-confidence",
        array=output_pdac_confidence,
    )

    return 0


def perform_inference(input_image, input_metadata):    
    # Define nnUNet v2 model parameters
    model_name = '3d_fullres'
    trainer_class_name = 'CustomTrainer'
    plans_identifier = 'nnUNetPlans'
    task_name = 'CURVASPDAC'

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
            use_folds=("0",),
            checkpoint_name="checkpoint_best.pth",
    )
    
    input_metadata = [input_metadata[i] for i in [2,0,1]]

    output_segmentation, probabilities = predictor.predict_single_npy_array(input_image=np.expand_dims(input_image, axis=0), 
                                                                            image_properties={'spacing':input_metadata},
                                                                            segmentation_previous_stage=None, 
                                                                            output_file_truncated=None, 
                                                                            save_or_return_probabilities=True
                                                                            )

    print('output_segmentation: '+str(output_segmentation.shape))
    print('probabilities: '+str(probabilities[1].shape))
    
    return output_segmentation, probabilities[1]


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result), result.GetSpacing()


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    # You may need to change the suffix to .tif to match the expected output
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
