#!/bin/bash

python initial_alignment.py --fixed "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD121759/slab08S/LFBCV_43734_1000.nii.gz" \
 --moving "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD121759/INDD121759_reslice.nii.gz" \
 --brainmold_path "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD121759/brainmold_widen_dots" \
 --working_dir "/Users/cathalye/Projects/histo_mri_INR/data_affine/INDD121759/work/slab08S" \
 --slab_num 08 \
 --overwrite
