import os
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil


def create_dummy_modalities(input_file_path, number_of_modalities, output_path="/root/breast_seg/dummy_modalities"):
    """
    Create multiple dummy modalities by copying a single image file.
    Saves to {output_path}/{filename_without_extension}/{filename_without_extension}_{modality_number}.nii.gz
    
    Args:
        input_file_path: Path to the input image file (e.g., .nii.gz or .nii)
        number_of_modalities: Number of dummy modalities to create
        output_path: Base output directory path
    
    Returns:
        List of paths to the output files
    """
    input_path = Path(input_file_path)
    
    # Extract file name without extension
    file_name_without_extension = input_path.stem
    if input_path.suffix == '.gz':
        # Handle .nii.gz case
        file_name_without_extension = Path(input_path.stem).stem
    
    # Create output directory structure
    output_dir = Path(output_path) / file_name_without_extension
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    
    # Create dummy modalities
    for modality_num in range(number_of_modalities):
        output_file = output_dir / f"{file_name_without_extension}_{modality_num:04d}.nii.gz"
        
        # Copy the file
        shutil.copy2(input_path, output_file)
        print(f"Created dummy modality {modality_num}: {output_file}")
        output_files.append(output_file)
    
    return output_files