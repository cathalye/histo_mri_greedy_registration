"""
Registration script for intial alignment of postmortem MRI to histology.

Takes as inputs:
(1) Reference histology image (assumed LFB-CV)
(2) Postmortem hemisphere MRI as the moving image
(3) Slab number
(4) Work directory for saving intermediate files

Processes:
(1) Extract the MRI slab corresponding to the histology slide
(2) Preprocess histology
(3) Initial alignment of the MRI slab to the histology slide

Outputs:
(1) ITK snap workspace for manual alignment
"""

import argparse
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.histology_data import HistologyData
from src.mri_data import MRIData
from src.histo_mri_registration import HistoMRIRegistration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=str, required=True, help="Path to fixed image (histology)")
    parser.add_argument("--moving", type=str, required=True, help="Path to moving image (MRI)")
    parser.add_argument("--brainmold_path", type=str, required=True, help="Path to brainmold slabs")
    parser.add_argument("--slab_num", type=int, required=True, help="Slab number")
    parser.add_argument("--working_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing files")
    args = parser.parse_args()

    slab_num = args.slab_num
    working_dir = args.working_dir
    brainmold_path = args.brainmold_path
    overwrite = args.overwrite

    # Create output directory if it doesn't exist
    os.makedirs(working_dir, exist_ok=True)
    # Create a subdirectory for the histology data
    histology_path = os.path.join(working_dir, "histology")
    os.makedirs(histology_path, exist_ok=True)
    # Create a subdirectory for the manual initialization
    manual_init_path = os.path.join(working_dir, "manual")
    os.makedirs(manual_init_path, exist_ok=True)
    # Create a subdirectory for the transforms
    transforms_path = os.path.join(working_dir, "transforms")
    os.makedirs(transforms_path, exist_ok=True)
    # Create a subdirectory for the MRI data
    mri_path =os.path.join(working_dir, "mri")
    os.makedirs(mri_path, exist_ok=True)

    # Load data
    mri_data = MRIData(args.moving)
    histology_data = HistologyData(args.fixed)

    # Extract the slab from whole hemisphere MRI
    mri_slab_path = os.path.join(mri_path, f"mri_slab.nii.gz")
    # mri_data.get_mri_slab(slab_num, save_path=mri_slab_path, return_img=False)
    mri_data.get_mri_slab_from_brainmold(slab_num, brainmold_path=brainmold_path, save_path=mri_slab_path, overwrite=overwrite)

    # Preprocess histology
    histology_data.preprocess_histology(channel=0, base_dir=histology_path, overwrite=overwrite)

    # Create registration processor
    processor = HistoMRIRegistration(working_dir)

    # Create initial alignment using image centers
    processor.create_initial_alignment(overwrite=overwrite)

    # Save manual itksnap workspace
    print(f"Saving manual itksnap workspace...")
    processor.save_manual_itksnap_workspace(overwrite=overwrite)
