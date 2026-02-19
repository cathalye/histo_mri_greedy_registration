#!/bin/bash

DATA_DIR="/Users/cathalye/Desktop/data_shortlist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find all INDD directories
INDD_DIRS=$(find "$DATA_DIR" -maxdepth 1 -type d -name "INDD*" | sort)

INITIAL="purple"
if [[ -z "$INDD_DIRS" ]]; then
    echo "No INDD directories found in $DATA_DIR"
    exit 1
fi

# Process each INDD directory
for indd_dir in $INDD_DIRS; do
    indd_name=$(basename "$indd_dir")
    echo "=========================================="
    echo "Processing INDD directory: $indd_name"
    echo "=========================================="

    # Find all work subdirectories (slab directories)
    work_dir="$indd_dir/work"
    if [[ ! -d "$work_dir" ]]; then
        echo "  Warning: No work directory found for $indd_name"
        continue
    fi

    # Find all slab directories in work/
    slab_dirs=$(find "$work_dir" -maxdepth 1 -type d -name "slab*" | sort)

    if [[ -z "$slab_dirs" ]]; then
        echo "  Warning: No slab directories found in $work_dir"
        continue
    fi

    # Process each slab directory
    for slab_dir in $slab_dirs; do
        slab_name=$(basename "$slab_dir")
        echo ""
        echo "  Processing slab: $slab_name"
        echo "  Working directory: $slab_dir"

        # Check if manual initialization exists
        if [[ $INITIAL == "manual" ]]; then
            manual_init_path="$slab_dir/initialization/manual_init_result.itksnap"
            if [[ ! -f "$manual_init_path" ]]; then
                echo "    Warning: Manual initialization not found at $manual_init_path"
                echo "    Skipping this slab (run initial_alignment.py first)"
                continue
            fi
        elif [[ $INITIAL == "purple" ]]; then
            purple_rigid_transform_path="$slab_dir/initialization/purple_rigid.mat"
            if [[ ! -f "$purple_rigid_transform_path" ]]; then
                echo "    Warning: Purple rigid transform not found at $purple_rigid_transform_path"
                echo "    Skipping this slab (run initial_alignment.py first)"
                continue
            fi
        fi

        # Run greedy registration
        echo "    Running greedy_registration.py..."
        python "$SCRIPT_DIR/greedy_registration.py" \
            --working_dir "$slab_dir" \
            --initial $INITIAL \
            --overwrite

        if [[ $? -eq 0 ]]; then
            echo "    ✓ Successfully processed $slab_name"
        else
            echo "    ✗ Error processing $slab_name"
        fi
    done
done

echo ""
echo "=========================================="
echo "Batch processing complete!"
echo "=========================================="
