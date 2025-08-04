#!/bin/bash

python initial_alignment.py --fixed "/Users/cathalye/Projects/histo_mri_greedy_registration/data/slide_32161_raw_1000.nii.gz" \
 --moving "/Users/cathalye/Projects/histo_mri_greedy_registration/data/102373L_reslice.nii.gz" \
 --working_dir "/Users/cathalye/Projects/histo_mri_greedy_registration/data/work" \
 --slab_num 3
