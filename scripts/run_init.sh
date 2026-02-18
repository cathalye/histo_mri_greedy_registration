#!/bin/bash

python initial_alignment.py --fixed "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD116571/slab04S/LFBCV_51709_1000.nii.gz" \
 --moving "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD116571/INDD116571_reslice.nii.gz" \
 --purple_segmentation "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD116571/INDD116571_purple_segmentation.nii.gz" \
 --reslice_transform "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD116571/116571R_reslice.mat" \
 --brainmold_path "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD116571/brainmold_widen_dots" \
 --working_dir "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD116571/work/slab04S" \
 --slab_num 04 \
 --n_cuts 2 \
 --chosen_cut S \
 --overwrite
