from pathlib import Path

from glob import glob
import SimpleITK
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy.ndimage import zoom
import numpy as np
import os
import torch
import gc

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def run():
    input_thoracic_abdominal_ct_image_SITK, input_metadata = load_image_file_as_array(
        location=INPUT_PATH / "images/thoracic-abdominal-ct",
    )

    input_thoracic_abdominal_ct_image = SimpleITK.GetArrayFromImage(input_thoracic_abdominal_ct_image_SITK)

    print('input_thoracic_abdominal_ct_image: '+str(input_thoracic_abdominal_ct_image.shape)+' '+str(type(input_thoracic_abdominal_ct_image)))
    print('input_metadata: '+str(input_metadata))
    
    _show_torch_cuda_info()
    output_pdac_segmentation, output_pdac_confidence = perform_inference(
        input_image=input_thoracic_abdominal_ct_image, input_metadata=input_metadata
    )

    del input_thoracic_abdominal_ct_image, input_metadata
    gc.collect()

    print('output_pdac_segmentation: '+str(output_pdac_segmentation.shape)+' '+str(np.unique(output_pdac_segmentation)))
    print('output_pdac_confidence: '+str(output_pdac_confidence.shape)+' '+str(np.unique(output_pdac_confidence)))

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/pdac-segmentation",
        array=output_pdac_segmentation,
        original_ct=input_thoracic_abdominal_ct_image_SITK
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/pdac-confidence",
        array=output_pdac_confidence,
        original_ct=input_thoracic_abdominal_ct_image_SITK,
    )

    del output_pdac_segmentation, output_pdac_confidence
    torch.cuda.empty_cache()
    gc.collect()

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

    if input_image.shape[1] <= 512:
        print(f'prediction done in {torch.device}')
        output_segmentation, probabilities = predictor.predict_single_npy_array(input_image=np.expand_dims(input_image, axis=0), 
                                                                            image_properties={'spacing':input_metadata},
                                                                            segmentation_previous_stage=None, 
                                                                            output_file_truncated=None, 
                                                                            save_or_return_probabilities=True
                                                                            )
        
        output_segmentation = output_segmentation.astype(np.uint8)
        probability = probabilities[1].astype(np.float32)

        del probabilities
        gc.collect()

    else:
        # Downsample the input image and adjust spacing
        downsample_factor = 0.5
        original_shape = input_image.shape
        zoom_factors = (1.0, downsample_factor, downsample_factor)
        input_image_ds = zoom(input_image, zoom=zoom_factors, order=1)  #
        spacing_ds = [input_metadata[0], input_metadata[1] / downsample_factor, input_metadata[2] / downsample_factor]  # spacing increases as resolution decreases

        print(f'input_image_ds: {input_image_ds.shape}')

        #predictor.patch_size = (96, 192, 192)
        #print(predictor.patch_size)

        output_segmentation_ds, probabilities_ds = predictor.predict_single_npy_array(
            input_image=np.expand_dims(input_image_ds.astype(np.float32), axis=0),
            image_properties={'spacing': spacing_ds},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True
        )

        print(f'output_segmentation_ds: {output_segmentation_ds.shape}, {np.unique(output_segmentation_ds)}')
        print(f'probabilities_ds[1]: {probabilities_ds[1].shape}, {np.unique(probabilities_ds[1])}')

        # Upsample back to original Y, X (leave Z as is)
        zoom_back = (1.0,
                 original_shape[1] / output_segmentation_ds.shape[1],
                 original_shape[2] / output_segmentation_ds.shape[2])

        # Upsample the outputs back to the original shape
        output_segmentation = zoom(output_segmentation_ds.astype(np.uint8), zoom=zoom_back, order=0).astype(np.uint8)  # nearest for segmentation
        probability = zoom(probabilities_ds[1].astype(np.float32), zoom=zoom_back, order=0).astype(np.float32)  # linear for probability map

        print(f'output_segmentation: {output_segmentation.shape}, {np.unique(output_segmentation)}')
        print(f'probabilities_ds[1]: {probability.shape}, {np.unique(probability)}')

        del output_segmentation_ds, probabilities_ds
        gc.collect()

    print(f'output_segmentation: {output_segmentation.shape}, {np.unique(output_segmentation)}')
    print(f'probability: {probability.shape}, {np.unique(probability)}')

    torch.cuda.empty_cache()
    
    return output_segmentation, probability


def pad_to_divisible_by_32(image):
    import numpy as np

    def pad_to_32(x):
        pad = (32 - (x % 32)) % 32
        pad_before = pad // 2
        pad_after = pad - pad_before
        return pad_before, pad_after

    z_pad = pad_to_32(image.shape[0])
    y_pad = pad_to_32(image.shape[1])
    x_pad = pad_to_32(image.shape[2])

    padded = np.pad(image,
        pad_width=((z_pad[0], z_pad[1]),
                   (y_pad[0], y_pad[1]),
                   (x_pad[0], x_pad[1])),
        mode='constant',
        constant_values=0
    )

    return padded, (z_pad, y_pad, x_pad)


def unpad_prediction(padded_pred, pads):
    z_pad, y_pad, x_pad = pads
    return padded_pred[
        z_pad[0]: padded_pred.shape[0] - z_pad[1],
        y_pad[0]: padded_pred.shape[1] - y_pad[1],
        x_pad[0]: padded_pred.shape[2] - x_pad[1]
    ]


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


def write_array_as_image_file(*, location, array, original_ct):
    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"

    print(f'{array.shape}, {np.unique(array)}')

    image = SimpleITK.GetImageFromArray(array)

    image.CopyInformation(original_ct)
    print('spacing: '+str(image.GetSpacing()))
    
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
