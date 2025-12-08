import glob
import os
import nibabel as nib
import SimpleITK as sitk
from picsl_c3d import Convert3D

c3d = Convert3D()

class MRIData:
    """
    A class for processing postmortem MRI data and extracting slabs.

    This class provides functionality for:
    - Loading whole hemisphere postmortem MRI data
    - Determining anterior-posterior (AP) axis orientation
    - Calculating slab boundaries based on slab number
    - Extracting MRI slabs corresponding to histology slides

    The class handles different coordinate system orientations and automatically
    determines the correct AP axis direction for slab extraction.
    """

    def __init__(self, mri_path):
        """
        Initialize the MRI data processor.

        Parameters
        ----------
        mri_path : str
            Path to the whole hemisphere postmortem MRI file (NIfTI format).

        Notes
        -----
        Loads the MRI data using both SimpleITK and nibabel for different processing needs.
        Extracts key metadata including:
        - Image spacing (voxel dimensions)
        - Image size (number of voxels in each dimension)

        Automatically calculates AP axis information to determine the correct
        orientation for slab extraction.
        """
        self.mri_path = mri_path
        self.sitk_image = sitk.ReadImage(mri_path)
        self.nib_image = nib.load(mri_path)

        # Extract metadata
        self.spacing = self.sitk_image.GetSpacing()
        self.size = self.sitk_image.GetSize()

        # Calculate AP axis information
        self._calculate_ap_axis()


    def _calculate_ap_axis(self):
        """
        Determine the anterior-posterior (AP) axis and its direction.

        This function analyzes the MRI image orientation to identify which axis
        corresponds to the anterior-posterior direction and whether it goes from
        anterior to posterior or posterior to anterior.

        Notes
        -----
        Uses nibabel's orientation analysis to determine the coordinate system.
        Sets two key attributes:
        - ap_axis: Index of the axis corresponding to AP direction (0, 1, or 2)
        - ap_direction: Direction of the AP axis (1 for A->P, -1 for P->A)

        This information is critical for correctly extracting slabs that correspond
        to specific histology slides.
        """
        affine = self.nib_image.affine
        orientation = nib.orientations.aff2axcodes(affine)

        self.ap_axis = orientation.index('A') if 'A' in orientation else orientation.index('P')
        self.ap_direction = 1 if orientation[self.ap_axis] == 'A' else -1


    def _get_start_end_mm(self, slab_num):
        """
        Calculate the start and end positions in millimeters for a given slab.

        Parameters
        ----------
        slab_num : int
            Slab number (1-based indexing).

        Returns
        -------
        tuple
            (start_mm, end_mm) - Start and end positions in millimeters.

        Notes
        -----
        Assumes each slab is 10 mm thick with 2 mm spacing between slabs.
        The formula used is:
        - start_mm = slab_num * 12 - 4
        - end_mm = start_mm + 10

        This accounts for the physical layout of the cutting mold used for
        histology sampling.
        """
        # Assume each slab is 10 mm thick and each slit in the mold is 2 mm
        start_mm = slab_num * 12 - 4
        end_mm = start_mm + 10
        return start_mm, end_mm


    def _get_slab_start_end_voxels(self, slab_num):
        """
        Convert millimeter positions to voxel indices for slab extraction.

        Parameters
        ----------
        slab_num : int
            Slab number (1-based indexing).

        Returns
        -------
        tuple
            (start_voxel, end_voxel) - Start and end voxel indices.

        Raises
        ------
        AssertionError
            If start_voxel >= end_voxel, indicating invalid slab boundaries.

        Notes
        -----
        This function handles different AP axis orientations:

        - If ap_direction == 1 (A->P): Voxel 0 is at the anterior end
        - If ap_direction == -1 (P->A): Voxel 0 is at the posterior end

        The conversion from millimeters to voxels accounts for the image spacing
        and the direction of the AP axis to ensure correct slab extraction
        regardless of the original image orientation.
        """
        start_mm, end_mm = self._get_start_end_mm(slab_num)

        if self.ap_direction == 1:
            # i.e. the AP axis goes from posterior to anterior
            # so the voxel 0 is the posterior end of the image
            ap_size = self.size[self.ap_axis] * self.spacing[self.ap_axis]  # total size of the hemisphere in mm in AP direction
            end_voxel = int((ap_size - start_mm) / self.spacing[self.ap_axis])
            start_voxel = int((ap_size - end_mm) / self.spacing[self.ap_axis]) - 1
        elif self.ap_direction == -1:
            # i.e. the AP axis goes from anterior to posterior
            # so the voxel 0 is the anterior end of the image
            start_voxel = int(start_mm / self.spacing[self.ap_axis])
            end_voxel = int(end_mm / self.spacing[self.ap_axis]) + 1

        assert start_voxel < end_voxel, "Start voxel must be less than end voxel"

        return start_voxel, end_voxel


    def get_mri_slab(self, slab_num, save_path=None, return_img=True):
        """
        Extract an MRI slab corresponding to a specific histology slide.

        Parameters
        ----------
        slab_num : int
            Slab number (1-based indexing) corresponding to the histology slide.
        save_path : str, optional
            Path to save the extracted slab. If None, slab is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.

        Returns
        -------
        SimpleITK.Image or None
            The extracted MRI slab if return_img=True, otherwise None.

        Notes
        -----
        This function extracts a 3D slab from the whole hemisphere MRI that
        corresponds to a specific histology slide. The extraction process:

        1. Calculates the physical boundaries of the slab in millimeters
        2. Converts these boundaries to voxel indices based on image spacing
        3. Accounts for the AP axis orientation and direction
        4. Extracts the region of interest using SimpleITK's RegionOfInterest

        The extracted slab maintains the original image metadata (spacing, origin,
        direction) and can be used directly in the registration pipeline.
        """
        start_voxel, end_voxel = self._get_slab_start_end_voxels(slab_num)

        # set the index of slice to be extracted in AP axis to the start voxel
        index = [0, 0, 0]
        index[self.ap_axis] = start_voxel

        slice_size = list(self.size)
        slice_size[self.ap_axis] = end_voxel - start_voxel + 1

        # use the functional interface to extract the slice
        sliced_mri = sitk.RegionOfInterest(self.sitk_image, size=slice_size, index=index)

        if save_path:
            sitk.WriteImage(sliced_mri, save_path)

        if return_img:
            return sliced_mri
        else:
            return None


    def get_mri_slab_from_brainmold(self, slab_num, brainmold_path, save_path):
        """
        Extract MRI slab corresponding to the brainmold slab.

        The above functions are "logically correct". However, there is no guarantee that the
        MRI mask is cropped tightly that 0 corresponds to the first voxel of the MRI slab.

        Therefore, using the brainmold slab is a more robust way to extract the MRI slab.

        Further, we pad the brainmold slab by 10 voxels in both anterior and posterior directions to
        account for any out of plane deformations while sectioning the histology tissue.

        Parameters
        ----------
        slab_num : int
            Slab number corresponding to the histology slide.
        brainmold_path : str
            Path to the brainmold slabs.
        save_path : str
            Path to save the extracted MRI slab.

        """
        slab_path = os.path.join(brainmold_path, f"*_slab{slab_num:02d}_mask_with_dots.nii.gz")
        slab_file = glob.glob(slab_path)[0]

        c3d.execute(f'{slab_file} -pad 0x10x0 0x10x0 0')

        padded_slab = c3d.pop()
        sitk.WriteImage(padded_slab, "tmp_padded_slab.nii.gz")
        c3d.execute(f'tmp_padded_slab.nii.gz {self.mri_path} -reslice-identity')

        extracted_slab = c3d.pop()

        if save_path:
            sitk.WriteImage(extracted_slab, save_path)

        os.remove("tmp_padded_slab.nii.gz")
