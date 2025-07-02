"""
Registration script for postmortem MRI to histology.

Takes as inputs:
(1) LFB-CV slide as the fixed image
(2) Postmortem hemisphere MRI as the moving image
(3) Slab number

Processes:
(1) Mask of the histology slide
(2) Extract the MRI slab corresponding to the histology slide
(3) Initial alignment of the MRI slab to the histology slide

Outputs:
(1) ITK snap workspace for manual alignment
"""

import argparse
import sys
import os

import SimpleITK as sitk

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mri_data import MRIData
from histology_data import HistologyData
from registration import HistoMRIRegistration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=str, required=True, help="Path to fixed image (histology)")
    parser.add_argument("--moving", type=str, required=True, help="Path to moving image (MRI)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the slab")
    parser.add_argument("--slab_num", type=int, required=True, help="Slab number")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    # Create a subdirectory for the histology data
    histology_path = os.path.join(args.output_path, "histology")
    os.makedirs(histology_path, exist_ok=True)
    # Create a subdirectory for the manual initialization
    manual_init_path = os.path.join(args.output_path, "manual_initialization")
    os.makedirs(manual_init_path, exist_ok=True)

    # Load data
    print(f"Loading MRI data...")
    mri_data = MRIData(args.moving)

    print(f"Loading histology data...")
    histology_data = HistologyData(args.fixed)

    # Create registration processor
    processor = HistoMRIRegistration(mri_data, histology_data, args.slab_num)

    # Extract the slab and create initial alignment
    print(f"Extracting MRI slab {args.slab_num}...")
    mri_slab = mri_data.get_mri_slab(args.slab_num)
    slab_num = str(args.slab_num).zfill(2)
    mri_slab_path = os.path.join(args.output_path, f"mri_slab_{slab_num}.nii.gz")
    sitk.WriteImage(mri_slab, mri_slab_path)

    # Get single channel histology
    print(f"Getting single channel histology...")
    histo_single_channel = histology_data.get_single_channel_image(channel=1)
    historef_single_channel_path = os.path.join(histology_path, "historef_single_channel.nii.gz")
    sitk.WriteImage(histo_single_channel, historef_single_channel_path)

    # Get resampled histology
    print(f"Resampling histology to MRI resolution...")
    histo_resampled = histology_data.resample_to_mri_resolution()
    historef_resampled_path = os.path.join(histology_path, "historef_resampled.nii.gz")
    sitk.WriteImage(histo_resampled, historef_resampled_path)

    # Get binary mask
    print(f"Getting binary mask...")
    histo_binary = histology_data.get_binary_mask()
    historef_binary_mask_path = os.path.join(histology_path, "historef_binary_mask.nii.gz")
    sitk.WriteImage(histo_binary, historef_binary_mask_path)

    # Get lowres mask
    print(f"Getting lowres mask...")
    histo_lowres = histology_data.get_lowres_mask(histo_binary, histo_resampled)
    historef_lowres_mask_path = os.path.join(histology_path, "historef_lowres_mask.nii.gz")
    sitk.WriteImage(histo_lowres, historef_lowres_mask_path)

    # Get chunk mask
    # XXX: using file paths instead of images because using image_graph_cut in subprocess
    print(f"Getting chunk mask...")
    historef_chunk_mask_path = os.path.join(histology_path, "historef_chunk_mask.nii.gz")
    histo_chunk = histology_data.get_chunk_mask(historef_binary_mask_path, historef_chunk_mask_path)

    # Create initial alignment
    print(f"Creating initial alignment...")
    init_mat_path = os.path.join(args.output_path, "init.mat")
    processor.create_initial_alignment(init_mat_path)

    # Save manual itksnap workspace
    print(f"Saving manual itksnap workspace...")
    workspace_path = os.path.join(manual_init_path, "manual_alignment_input.itksnap")
    processor.save_manual_itksnap_workspace(
        image1_path=historef_resampled_path,
        image2_path=mri_slab_path,
        workspace_path=workspace_path,
        segmentation=historef_lowres_mask_path,
        transform=init_mat_path,
    )
