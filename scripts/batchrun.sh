#!/bin/bash

DATA_DIR="/Users/cathalye/Projects/data_histo_mri/test_subject"
WORK_DIR="/Users/cathalye/Projects/proj_histo_mri_greedy_registration/work"

# Loop through all specimen directories
for specimen_dir in "$DATA_DIR"/INDD*/; do
    # Extract specimen name from directory path
    specimen=$(basename "$specimen_dir")

    # Skip if not a directory or if it's a hidden directory
    if [[ ! -d "$specimen_dir" ]] || [[ "$specimen" == .* ]]; then
        continue
    fi

    echo "Processing specimen: $specimen"

    # Find the MRI file (*_reslice.nii.gz)
    mri_file=$(find "$specimen_dir" -maxdepth 1 -name "*_reslice.nii.gz" -type f)
    if [[ -n "$mri_file" ]]; then
        mri_path="$mri_file"
    else
        echo "  Warning: No MRI file found for specimen $specimen"
        continue
    fi

    # Loop through slab directories within each specimen
    for slab_dir in "$specimen_dir"slab*/; do
        # Skip if not a directory
        if [[ ! -d "$slab_dir" ]]; then
            continue
        fi

        # Extract slab name from directory path
        slab_name=$(basename "$slab_dir")
        # Extract slab number from slab name (first two digits only)
        slab_number=$(echo "$slab_name" | sed 's/slab//' | sed 's/[^0-9]//g' | head -c 2)

        echo "Processing slab: $slab_name"

        # Find the histology file (LFBCV_*)
        histo_file=$(find "$slab_dir" -maxdepth 1 -name "LFBCV_*" -type f)
        if [[ -n "$histo_file" ]]; then
            histo_path="$histo_file"
        else
            echo "    Warning: No histology file found for slab $slab_name"
            continue
        fi

        output_path="$WORK_DIR/$specimen/$slab_name"

        python initial_alignment.py --fixed "$histo_path" --moving "$mri_path" --output_path "$output_path" --slab_num "$slab_number"
        
    done
done
