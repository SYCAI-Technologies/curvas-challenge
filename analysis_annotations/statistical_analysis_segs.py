import nibabel as nib
import os
import numpy as np
import argparse
import itertools
from statistical_metrics import dice_similarity, hausdorff_distance
from statistical_metrics import parallel_compute_fleiss_kappa, isolate_structure_voxels_list
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import gc
import multiprocessing
import psutil

memory_info = psutil.virtual_memory()
total_memory = memory_info.total / (1024**3)  # Convert to GB
available_memory = memory_info.available / (1024**3)  # Convert to GB
print(f"Total memory: {total_memory:.2f} GB")
print(f"Available memory: {available_memory:.2f} GB")

num_cores = multiprocessing.cpu_count()
print(f"Number of CPU cores available: {num_cores}")


def get_args_parser():
    parser = argparse.ArgumentParser(description='Statistical Analysis for CURVAS Challenge dataset')
    # data
    parser.add_argument('--folder', type=str, default='validation_set', help='')   
    parser.add_argument('--path', type=str, default='validation_set', help='')   

    args = parser.parse_args()

    return args


def check_if_study_done(study_name, excel_file):
    if not os.path.exists(excel_file):
        return False  # If file doesn't exist, we haven't computed anything yet

    # Load all relevant sheets
    xls = pd.ExcelFile(excel_file)
    required_sheets = ['DSC', 'Hausdorff', 'CohenKappa', 'FleissKappa']
    
    # Check if all required sheets are present
    for sheet in required_sheets:
        if sheet not in xls.sheet_names:
            return False

    # Check for the study in all sheets
    for sheet in required_sheets:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        if study_name not in df['study'].values:
            return False
    
    return True


def load_existing_data(output_excel_file, sheet_name):
    """Load the existing data from the Excel sheet."""
    if os.path.exists(output_excel_file):
        try:
            return pd.read_excel(output_excel_file, sheet_name=sheet_name)
        except ValueError:  # If the sheet doesn't exist
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def main(args):

    data_path = f'{arg.path}/{args.folder}/'
    output_excel_file = f'{arg.path}/metrics_results_final.xlsx'  # Path to the Excel file
    
    list_files = [d for d in os.listdir(data_path) if 'UKCHLL' in d]
    
    print('Files to bue analyzed: '+str(list_files))
        
    # List of structure names
    structures = {"Pancreas": 1, "Kidney": 2, "Liver": 3}
    
    # Initialize results to store in Excel
    dsc_data = []
    hd_data = []
    cohen_kappa_data = []
    fleiss_kappa_data = []
    icc_data = []
    
    for count, study in enumerate(list_files, 1):
        print(f'{count}/{len(list_files)}- {study}')
        
        # Skip the study if it's already done
        if check_if_study_done(study, output_excel_file):
            print(f"Study {study} already processed, skipping...")
            continue
    
        annotation_1 = np.round(nib.load(os.path.join(data_path, study, 'annotation_1.nii.gz')).get_fdata()).astype(np.uint8)
        annotation_2 = np.round(nib.load(os.path.join(data_path, study, 'annotation_2.nii.gz')).get_fdata()).astype(np.uint8)
        annotation_3 = np.round(nib.load(os.path.join(data_path, study, 'annotation_3.nii.gz')).get_fdata()).astype(np.uint8)
        
        segmentations_by_structure = {}
        
        # Segment the organs
        for structure, i in structures.items():
            segmentations_by_structure[structure] = [
                [np.where(annotation_1 == i, 1, 0).astype(np.uint8),
                 np.where(annotation_2 == i, 1, 0).astype(np.uint8),
                 np.where(annotation_3 == i, 1, 0).astype(np.uint8)]
            ]
        
        del annotation_1, annotation_2, annotation_3
        gc.collect()
    
        # Compute DSC for each pair of raters for each structure, excluding background
        for structure, segmentations in segmentations_by_structure.items():
            
            for idx, ct_segmentations in enumerate(segmentations):  # Loop over CT scans
                # ct_segmentations = [CTN_ann1, CTN_ann2, CTN_ann3] (3 segmentations for CTN)
                
                print(f"Analyzing {structure} of study {study}")
            
                # Flatten the 3D segmentations for each rater
                flattened_segmentations = [seg.flatten() for seg in ct_segmentations]
                segmentations_isolated = isolate_structure_voxels_list(ct_segmentations)
                flattened_segmentations_isolated = [seg.flatten() for seg in segmentations_isolated]
                
                # ICC
                """print('Computing ICC...')
                icc_value = '%.4f'%parallel_compute_icc(flattened_segmentations_isolated)
                icc_data.append([study, args.folder, structure, float(icc_value)])
                print(f"ICC for {structure}, CT {idx + 1}: {icc_value}")"""
                
                # Fleiss Kappa
                print('Computing Fleiss Kappa...')
                fleiss_value = '%.4f'%parallel_compute_fleiss_kappa(flattened_segmentations_isolated)
                fleiss_kappa_data.append([study, args.folder, structure, float(fleiss_value)])
                print(f"Fleiss Kappa for {structure}, CT {idx + 1}: {fleiss_value}")
            
                # Compute Cohen's Kappa for each pair of raters
                pairs = list(itertools.combinations(range(len(ct_segmentations)), 2))
                
                print('pairs: '+str(pairs))
                
                for i, j in pairs:
                    
                    print(f'Pair: ({i}, {j})')
                    
                    # For each pair of segmentations (rater i and rater j)
                    seg_i = flattened_segmentations[i]
                    seg_j = flattened_segmentations[j]
                    
                    # DSC
                    print('Computing Dice Score...')
                    dsc_value = '%.4f'%(dice_similarity(seg_i, seg_j)*100)
                    dsc_data.append([study, args.folder, structure, f'Rater{i + 1}_vs_Rater{j + 1}', float(dsc_value)])
                    print(f"DSC between Rater {i + 1} and Rater {j + 1} for {structure}, CT {idx + 1}: {dsc_value}")

                    # Hausdorff Distance
                    print('Computing Hausdorff Distance...')
                    hd_value = '%.4f'%hausdorff_distance(seg_i.reshape(ct_segmentations[0].shape),
                                                         seg_j.reshape(ct_segmentations[0].shape))
                    hd_data.append([study, args.folder, structure, f'Rater{i + 1}_vs_Rater{j + 1}', float(hd_value)])
                    print(f"Hausdorff Distance between Rater {i + 1} and Rater {j + 1} for {structure}, CT {idx + 1}: {hd_value}")

                    # Cohen's Kappa
                    print('Computing Cohen.s Kappa...')
                    kappa = '%.4f'%cohen_kappa_score(seg_i, seg_j)
                    seg_i_struct, seg_j_struct = isolate_structure_voxels_list([seg_i, seg_j]) #isolate_structure_voxels(seg_i, seg_j)
                    kappa_isolated = '%.4f'%cohen_kappa_score(seg_i_struct, seg_j_struct) if len(seg_i_struct) > 0 and len(seg_j_struct) > 0 else np.nan
                    cohen_kappa_data.append([study, args.folder, structure, f"Rater{i+1}_vs_Rater{j+1}", float(kappa), float(kappa_isolated)])
                    print(f"Cohen's Kappa between Rater {i + 1} and Rater {j + 1} for {structure}, CT {idx + 1}: {kappa}")
                    print(f"Cohen's Kappa (isolated) between Rater {i + 1} and Rater {j + 1} for {structure}, CT {idx + 1}: {kappa_isolated}")
        
        # Save results to Excel
        with pd.ExcelWriter(output_excel_file, mode='a', if_sheet_exists='overlay') as writer:
            if dsc_data:
                existing_dsc = load_existing_data(output_excel_file, 'DSC')
                updated_dsc = pd.concat([existing_dsc, pd.DataFrame(dsc_data, columns=["study", "group", "structure", "rater_pair", "dsc"])])
                updated_dsc.to_excel(writer, sheet_name='DSC', index=False)

            if hd_data:
                existing_hd = load_existing_data(output_excel_file, 'Hausdorff')
                updated_hd = pd.concat([existing_hd, pd.DataFrame(hd_data, columns=["study", "group", "structure", "rater_pair", "hd"])])
                updated_hd.to_excel(writer, sheet_name='Hausdorff', index=False)

            if cohen_kappa_data:
                existing_ck = load_existing_data(output_excel_file, 'CohenKappa')
                updated_ck = pd.concat([existing_ck, pd.DataFrame(cohen_kappa_data, columns=["study", "group", "structure", "rater_pair", "kappa", "kappa_iso"])])
                updated_ck.to_excel(writer, sheet_name='CohenKappa', index=False)

            if fleiss_kappa_data:
                existing_fk = load_existing_data(output_excel_file, 'FleissKappa')
                updated_fk = pd.concat([existing_fk, pd.DataFrame(fleiss_kappa_data, columns=["study", "group", "structure", "fleiss_kappa"])])
                updated_fk.to_excel(writer, sheet_name='FleissKappa', index=False)
                #pd.DataFrame(icc_data, columns=["study", "group", "structure", "icc"]).to_excel(writer, sheet_name='ICC', index=False)
            
        # Clear the results after saving them to Excel to free memory
        dsc_data.clear()
        hd_data.clear()
        cohen_kappa_data.clear()
        fleiss_kappa_data.clear()
        
        gc.collect()

    print(f"All results saved to {output_excel_file}")
        
        
if __name__ == "__main__":
    args = get_args_parser()
    main(args)

    
    