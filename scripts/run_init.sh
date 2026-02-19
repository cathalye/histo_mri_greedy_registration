#!/bin/bash

python initial_alignment.py --fixed "/Users/cathalye/Desktop/data_shortlist/INDD119851/slab10S/LFBCV_36413_1000.nii.gz" \
 --moving "/Users/cathalye/Desktop/data_shortlist/INDD119851/INDD119851_reslice.nii.gz" \
 --purple_segmentation "/Users/cathalye/Desktop/data_shortlist/INDD119851/INDD119851_purple_segmentation.nii.gz" \
 --reslice_transform "/Users/cathalye/Desktop/data_shortlist/INDD119851/INDD119851_reslice.mat" \
 --brainmold_path "/Users/cathalye/Desktop/data_shortlist/INDD119851/brainmold" \
 --working_dir "/Users/cathalye/Desktop/data_shortlist/INDD119851/work/slab10S" \
 --slab_num 10 \
 --n_cuts 2 \
 --chosen_cut S \
 --overwrite
