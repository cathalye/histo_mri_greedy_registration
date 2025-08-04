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
    args = parser.parse_args()

    working_dir = args.working_dir

    # Check for manual initialization
    manual_init_path = os.path.join(working_dir, "manual/manual_init_result.itksnap")
    if not os.path.exists(manual_init_path):
        raise ValueError("Manual initialization not found. Please run initial_alignment.py first.")

    # Run registration
    processor = HistoMRIRegistration(working_dir)
    processor.extract_manual_alignment_matrix()
    processor.rigid_registration()
    processor.affine_registration()
    processor.deformable_registration()
    processor.apply_transforms()
