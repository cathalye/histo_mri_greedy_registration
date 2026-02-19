"""
Registration script for automatic postmortem MRI to histology registration.

Takes as inputs:
(1) Working directory
Assumes:
- Histology files are under working_dir/histology
- Manual alignment is under working_dir/manual_initialization

Processes:
(1) Extract the manual alignment
(2) Rigid registration
(3) Affine registration
(4) Deformable registration
(5) Apply all the transforms to the MRI slab

Outputs:
(1) ITK snap workspace for results
"""
import argparse
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.histo_mri_registration import HistoMRIRegistration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, required=True, help="Working directory")
    parser.add_argument("--initial", type=str, required=True, default="manual", help="Initial transform - manual or purple")
    parser.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing files")
    args = parser.parse_args()

    working_dir = args.working_dir
    initial = args.initial
    overwrite = args.overwrite

    # Check for manual initialization
    if initial == "manual":
        manual_init_path = os.path.join(working_dir, "initialization/manual_init_result.itksnap")
        if not os.path.exists(manual_init_path):
            raise ValueError("Manual initialization not found. Please run initial_alignment.py first.")
        processor = HistoMRIRegistration(working_dir, initial="manual")
        processor.extract_manual_alignment_matrix(overwrite=overwrite)
    elif initial == "purple":
        purple_rigid_transform_path = os.path.join(working_dir, "initialization/purple_rigid.mat")
        if not os.path.exists(purple_rigid_transform_path):
            raise ValueError("Purple rigid transform not found. Please run initial_alignment.py first.")
        processor = HistoMRIRegistration(working_dir, initial="purple")
    else:
        raise ValueError(f"Invalid initial transform: {initial}")

    # Run registration
    processor.rigid_registration(overwrite=overwrite)
    processor.affine_registration(overwrite=overwrite)
    processor.deformable_registration(overwrite=overwrite)
    processor.apply_transforms(overwrite=overwrite)
