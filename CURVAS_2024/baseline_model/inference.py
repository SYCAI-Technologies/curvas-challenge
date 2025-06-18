from pathlib import Path

from glob import glob
import time
import threading
import SimpleITK as sitk
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():

    input_thoracic_abdominal_ct_image, input_metadata = load_image_file_as_array(
        location=INPUT_PATH / "images/thoracic-abdominal-ct",
    )
    
    _show_torch_cuda_info()
    
    start_time = time.time()
        
    output_abdominal_organ_segmentation, output_pancreas_confidence, output_kidney_confidence, output_liver_confidence = perform_inference(
        input_image=input_thoracic_abdominal_ct_image, input_metadata=input_metadata
    )
    
    prediction_time = np.round((time.time() - start_time)/60, 2)
    print('Prediction time: '+str(prediction_time))
    
    print('Saving the predictions')
    
    start_time = time.time()
    
    write_files_in_parallel([(Path(OUTPUT_PATH / "images/abdominal-organ-segmentation"), output_abdominal_organ_segmentation),
                             (Path(OUTPUT_PATH / "images/kidney-confidence"), output_kidney_confidence),
                             (Path(OUTPUT_PATH / "images/pancreas-confidence"), output_pancreas_confidence),
                             (Path(OUTPUT_PATH / "images/liver-confidence"), output_liver_confidence)
                             ])
    
    saving_time = np.round((time.time() - start_time)/60, 2)
    print('Saving time: '+str(saving_time))
    
    print('Finished running algorithm!')
    
    return 0


def perform_inference(input_image, input_metadata):    
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
    
    input_metadata = [input_metadata[i] for i in [2,0,1]]

    output_segmentation, probabilities = predictor.predict_single_npy_array(input_image=np.expand_dims(input_image, axis=0), 
                                                                            image_properties={'spacing':input_metadata},
                                                                            segmentation_previous_stage=None, 
                                                                            output_file_truncated=None, 
                                                                            save_or_return_probabilities=True
                                                                            )
    
    return output_segmentation, probabilities[1],  probabilities[2], probabilities[3]


def load_image_file_as_array(*, location):
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    result = sitk.ReadImage(input_files[0])

    return sitk.GetArrayFromImage(result), result.GetSpacing()


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"

    image = sitk.GetImageFromArray(array)
    sitk.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )
    

def write_files_in_parallel(files_data):
    threads = []
    for location, array in files_data:
        print('location: '+str(location))
        thread = threading.Thread(target=write_array_as_image_file, kwargs={'location': location, 'array': array})
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


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
