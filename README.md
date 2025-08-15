# Postmortem MRI to Histology Registration

A comprehensive pipeline for registering postmortem MRI data to histology slides using the Greedy registration algorithm. This project implements a multi-stage registration approach that combines manual initialization with automated rigid, affine, and deformable registration steps.

The protocol for postmortem MRI imaging and subsequent histology sampling is described in detail in our manuscript - [Operationalizing postmortem pathology-MRI association studies in Alzheimer’s disease and related disorders with MRI-guided histology sampling](https://link.springer.com/article/10.1186/s40478-025-02030-y). This code implements the histology-MRI registration pipeline presented in the paper.

## Features

- **Multi-stage Registration**: Initial manual alignment followed by automated rigid, affine, and deformable registration
- **MRI Slab Extraction**: Automatic extraction of MRI slabs corresponding to specific histology slides
- **Histology Preprocessing**: Resampling and smoothing of histology data to match MRI resolution
- **ITK-SNAP Initialization**: Workspace generation for manual alignment and result visualization
- **Batch Processing**: Support for processing multiple specimens and slabs

## Installation

### Prerequisites

- Python 3.10
- Conda or Miniconda
- ITK-SNAP (for manual alignment)
- Convert3D (c3d) command-line tools

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd histo_mri_greedy_registration
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate greedy_registration
```

1. Install ITK-SNAP (www.itksnap.org)

## Usage

Documentation for
   - c3d: https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md
   - greedy: https://sites.google.com/view/greedyreg/documentation

The python attributes correspond to the CLI attributes described above.

The registration pipeline consists of three main steps:

### 1. Initial Alignment

```bash
python scripts/initial_alignment.py \
    --fixed /path/to/histology.nii.gz \
    --moving /path/to/mri.nii.gz \
    --slab_num 3 \
    --working_dir /path/to/working/directory
```

This step:
- Extracts the MRI slab corresponding to the specified slab number
- Preprocesses the histology data (resampling, smoothing)
- Creates initial center-based alignment
- Generates an ITK-SNAP workspace for manual refinement

### 2. Manual Alignment

- Open the ITK-SNAP workspace `manual_init.itksnap` saved in step 1
- Go to Tools > Image Registration
  - Set the scaling of all axes to 1.1 (this accounts for the shrinkage of tissue in fixation and is usually the best ratio, but may vary on a case-by-case basis)
  - Choose Interactive Tool
    - Move around the MRI in all Axial, Sagittal, and Coronal planes to best align it with the histology slide
    - You can toggle the segmentation mask by pressing S key on your keyboard
    - Given the manual cutting process, the histology slide may be in an oblique plane. You can use an oblique plane by adjusting the interactive circle
- Once you're satisfied with the manual alignment, save the workspace by navigating to
  - Workspace > Save Workspace As > `manual_init_result.itksnap`
  - Make sure you save the workspace in the same directory as the original workspace and that it contains the suffix `_result`

### 3. Registration Pipeline

After manual alignment in ITK-SNAP, run the automated registration:

```bash
python scripts/greedy_registration.py \
    --working_dir /path/to/working/directory
```

This step:
- Extracts the manual alignment transform
- Performs rigid registration
- Performs affine registration
- Performs deformable registration
- Applies all transforms and generates result workspace

### Working Directory Structure

After running the registration pipeline, the working directory will contain:

```
working_dir/
├── mri_slab.nii.gz             # Extracted MRI slab
├── histology/                  # Histology processing outputs
│   ├── historef_single_channel.nii.gz  # Single channel extracted histology
│   ├── historef_resampled.nii.gz       # Histology resampled to MRI resolution
│   ├── historef_binary_mask.nii.gz     # Binary mask from Otsu thresholding
│   └── historef_lowres_mask.nii.gz     # Low-resolution mask for registration
├── manual/                     # Manual alignment files
│   ├── manual_init.itksnap     # Initial ITK-SNAP workspace
│   └── manual_init_result.itksnap # Manual alignment result
├── transforms/                 # Registration transforms
│   ├── centers_init.mat        # Initial center alignment
│   ├── manual_init.mat         # Manual alignment transform
│   ├── rigid.mat               # Rigid registration transform
│   ├── affine.mat              # Affine registration transform
│   └── deformable.mhd          # Deformable registration transform
├── *_result.nii.gz             # Registered MRI at each stage
│   ├── manual_result.nii.gz    # After manual alignment
│   ├── rigid_result.nii.gz     # After rigid registration
│   ├── affine_result.nii.gz    # After affine registration
│   └── deformable_result.nii.gz # After deformable registration
└── results.itksnap             # Final ITK-SNAP workspace with all results
```

### Batch Processing

Use the batch processing script for multiple specimens:

```bash
bash scripts/batchrun.sh
```

Modify the script to specify your data and working directories structure.

## Input Data Requirements

### MRI Data
- Whole hemisphere postmortem MRI in NIfTI format
- Should be resliced to consistent orientation
- File naming convention: `*_reslice.nii.gz`

### Histology Data
- Histology slides in NIfTI format
- Multi-channel images (e.g., LFB-CV stained)
- File naming convention: `LFBCV_*` or `slide_*_raw_*.nii.gz`

## Output Files

The registration pipeline generates several output files in the working directory:

- `mri_slab.nii.gz` - Extracted MRI slab
- `histology/historef_resampled.nii.gz` - Preprocessed histology
- `histology/historef_lowres_mask.nii.gz` - Histology mask
- `transforms/` - All registration transforms
- `*_result.nii.gz` - Registered MRI at each stage
- `results.itksnap` - ITK-SNAP workspace with all results

## Dependencies

### Python Packages
- `nibabel` (5.3.2) - Neuroimaging file I/O
- `numpy` (1.26.4) - Numerical computing
- `SimpleITK` (2.5.2) - Medical image processing
- `picsl_c3d` (1.4.2.2) - Convert3D Python interface
- `picsl_greedy` (0.0.6) - Greedy registration Python interface

### External Tools
- **ITK-SNAP** - Manual alignment and visualization
- **Convert3D (c3d)** - Image processing utilities
- **Greedy** - Registration algorithm

## Registration Pipeline Details

1. **Initial Alignment**: Center-based alignment using image centroids
2. **Manual Refinement**: User-guided alignment in ITK-SNAP
3. **Rigid Registration**: 6-DOF rigid body transformation
4. **Affine Registration**: 12-DOF affine transformation
5. **Deformable Registration**: Non-linear deformation field

Each stage uses the Weighted Normalized Cross-Correlation (WNCC) similarity metric and includes appropriate regularization.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all external tools (ITK-SNAP, c3d) are properly installed
2. **File Paths**: Use absolute paths or ensure relative paths are correct
3. **Memory Issues**: Large images may require increased memory allocation
4. **Manual Alignment**: Ensure manual alignment is completed before running registration

### Error Messages

- "Manual initialization not found": Run initial alignment first
- "No MRI file found": Check file naming and directory structure
- "No histology file found": Verify histology file exists and naming convention

## Citation

If you use this software in your research, please cite:

```bibtex
@article{athalye2025,
  title = {Operationalizing Postmortem Pathology-MRI Association Studies in Alzheimer’s Disease and Related Disorders with MRI-guided Histology Sampling},
  author = {Athalye, Chinmayee and Bahena, Alejandra and Khandelwal, Pulkit and Emrani, Sheina and Trotman, Winifred and Levorse, Lisa M. and Khodakarami, Zahra and Ohm, Daniel T. and Teunissen-Bermeo, Eric and Capp, Noah and Sadaghiani, Shokufeh and Arezoumandan, Sanaz and Lim, Sydney A. and Prabhakaran, Karthik and Ittyerah, Ranjit and Robinson, John L. and Schuck, Theresa and Lee, Edward B. and Tisdall, M. Dylan and Das, Sandhitsu R. and Wolk, David A. and Irwin, David J. and Yushkevich, Paul A.},
  date = {2025-05-28},
  journaltitle = {Acta Neuropathologica Communications},
  volume = {13},
  number = {1},
  pages = {120},
  issn = {2051-5960},
  doi = {10.1186/s40478-025-02030-y},
  url = {https://doi.org/10.1186/s40478-025-02030-y},
}
```
