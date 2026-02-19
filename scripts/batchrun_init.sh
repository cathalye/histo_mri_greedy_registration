#!/bin/bash

DATA_DIR="/Users/cathalye/Desktop/data_shortlist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all INDD directories
INDD_DIRS=$(find "$DATA_DIR" -maxdepth 1 -type d -name "INDD*" | sort)

if [[ -z "$INDD_DIRS" ]]; then
    echo "No INDD directories found in $DATA_DIR"
    exit 1
fi

for indd_dir in $INDD_DIRS; do
    specimen=$(basename "$indd_dir")
    echo "Processing specimen: $specimen"

    # Moving image: INDD/*_reslice.nii.gz
    mri_file=$(find "$indd_dir" -maxdepth 1 -name "*_reslice.nii.gz" -type f | head -1)
    if [[ -z "$mri_file" ]]; then
        echo "  Warning: No MRI file (*_reslice.nii.gz) found for $specimen"
        continue
    fi
    mri_path="$mri_file"

    # Purple segmentation: INDD/*_segmentation.nii.gz
    purple_seg_file=$(find "$indd_dir" -maxdepth 1 -name "*_segmentation.nii.gz" -type f | head -1)
    if [[ -z "$purple_seg_file" ]]; then
        echo "  Warning: No purple segmentation file (*_segmentation.nii.gz) found for $specimen"
        continue
    fi

    # Reslice transform: INDD/*_reslice.mat
    reslice_transform_file=$(find "$indd_dir" -maxdepth 1 -name "*_reslice.mat" -type f | head -1)
    if [[ -z "$reslice_transform_file" ]]; then
        echo "  Warning: No reslice transform file (*_reslice.mat) found for $specimen"
        continue
    fi

    # Brainmold path: INDD*/brainmold*
    brainmold_path="$indd_dir/brainmold"*
    if ! compgen -G "$indd_dir/brainmold"* > /dev/null 2>&1; then
        echo "  Warning: No brainmold path found for $specimen"
        continue
    fi

    # Find all slab directories under this INDD
    slab_dirs=$(find "$indd_dir" -maxdepth 1 -type d -name "slab*" | sort)
    if [[ -z "$slab_dirs" ]]; then
        echo "  Warning: No slab directories found for $specimen"
        continue
    fi

    for slab_dir in $slab_dirs; do
        slab_name=$(basename "$slab_dir")
        # slab_spec: e.g. slab09S -> 09S
        slab_spec="${slab_name#slab}"
        # Extract slab number (digits only)
        slab_number=$(echo "$slab_spec" | sed 's/[^0-9]//g')

        # Determine n_cuts and chosen_cut based on whether slab_spec ends with S or I
        last_char="${slab_spec: -1}"
        if [[ "$last_char" == "S" || "$last_char" == "s" ]]; then
            chosen_cut="S"
            n_cuts=2
        elif [[ "$last_char" == "I" || "$last_char" == "i" ]]; then
            chosen_cut="I"
            n_cuts=2
        else
            # No letter suffix means single cut
            chosen_cut="S"
            n_cuts=1
        fi

        echo "  Processing slab: $slab_name (n_cuts: $n_cuts, chosen_cut: $chosen_cut)"

        # Fixed image: INDD*/slab*/LFBCV*
        histo_file=$(find "$slab_dir" -maxdepth 1 -name "LFBCV_*" -type f | head -1)
        if [[ -z "$histo_file" ]]; then
            echo "    Warning: No histology file (LFBCV_*) found for slab $slab_name"
            continue
        fi
        histo_path="$histo_file"

        # Working dir: INDD*/work/slab*
        work_dir="$indd_dir/work/$slab_name"
        mkdir -p "$work_dir"

        python "$SCRIPT_DIR/initial_alignment.py" --fixed "$histo_path" \
            --moving "$mri_path" \
            --purple_segmentation "$purple_seg_file" \
            --reslice_transform "$reslice_transform_file" \
            --brainmold_path "$brainmold_path" \
            --working_dir "$work_dir" \
            --slab_num "$slab_number" \
            --n_cuts "$n_cuts" \
            --chosen_cut "$chosen_cut" \
            --overwrite
    done
done
