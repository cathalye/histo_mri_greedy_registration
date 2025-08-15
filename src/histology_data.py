import os

import numpy as np
import SimpleITK as sitk
from picsl_c3d import Convert3D

c3d = Convert3D()

class HistologyData:
    """
    A class for processing and preprocessing histology image data.

    This class provides functionality for:
    - Loading multi-channel histology images
    - Extracting single channels
    - Resampling to target resolution
    - Creating binary masks for registration
    - Complete preprocessing pipeline

    The class supports both SimpleITK and Convert3D (c3d) processing methods.
    """

    def __init__(self, histology_path):
        """
        Initialize the histology data processor.

        Parameters
        ----------
        histology_path : str
            Path to the histology image file (typically multi-channel NIfTI format).

        Notes
        -----
        Loads the histology image and extracts key metadata including:
        - Image spacing (voxel dimensions)
        - Image size (number of voxels in each dimension)
        - Image origin (physical coordinates of the first voxel)
        - Image direction (coordinate system orientation)
        """
        self.histology_path = histology_path
        self.sitk_image = sitk.ReadImage(histology_path)

        # Extract metadata
        self.spacing = self.sitk_image.GetSpacing()
        self.size = self.sitk_image.GetSize()
        self.origin = self.sitk_image.GetOrigin()
        self.direction = self.sitk_image.GetDirection()


    def get_single_channel_image(self, channel=1, save_path=None, return_img=True):
        """
        Extract a single channel from a multi-channel histology image.

        Parameters
        ----------
        channel : int, optional
            Channel index to extract (0-based indexing). Default is 1.
        save_path : str, optional
            Path to save the extracted channel image. If None, image is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.

        Returns
        -------
        SimpleITK.Image or None
            The extracted single-channel image if return_img=True, otherwise None.

        Notes
        -----
        Uses SimpleITK's VectorIndexSelectionCast to extract the specified channel
        from the multi-channel input image. This is useful for histology images
        that contain multiple staining channels (e.g., LFB-CV staining).
        """
        single_channel_image = sitk.VectorIndexSelectionCast(self.sitk_image, channel)

        if save_path is not None:
            sitk.WriteImage(single_channel_image, save_path)

        if return_img:
            return single_channel_image
        else:
            return None


    def c3d_get_single_channel_image(self, channel=1, save_path=None, return_img=True):
        """
        Extract a single channel using Convert3D (c3d) tools.

        Parameters
        ----------
        channel : int, optional
            Channel index to extract (0-based indexing). Default is 1.
        save_path : str, optional
            Path to save the extracted channel image. If None, image is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.

        Returns
        -------
        SimpleITK.Image or None
            The extracted single-channel image if return_img=True, otherwise None.

        Notes
        -----
        This is an alternative to get_single_channel_image() using c3d tools.
        """
        # TODO: ask Paul why -mcs throws an error
        # Also had the same issue with data_histo_mri_paired_png
        # used subprocess to get around it there
        c3d.push(self.sitk_image)
        c3d.execute(f'-mcs -pick {channel}')

        single_channel_image = c3d.pop()

        if save_path is not None:
            sitk.WriteImage(single_channel_image, save_path)

        if return_img:
            return single_channel_image
        else:
            return None


    def resample_to_mri_resolution(self, channel=1, spacing=[0.3, 0.3, 1.0], save_path=None, return_img=True):
        """
        Resample histology image to MRI resolution with preprocessing.

        Parameters
        ----------
        channel : int, optional
            Channel index to extract and process. Default is 1.
        spacing : list, optional
            Target voxel spacing in millimeters [x, y, z]. Default is [0.3, 0.3, 1.0].
        save_path : str, optional
            Path to save the resampled image. If None, image is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.

        Returns
        -------
        SimpleITK.Image or None
            The resampled image if return_img=True, otherwise None.

        Notes
        -----
        This function performs a complete preprocessing pipeline:
        1. Extracts the specified channel from multi-channel input
        2. Applies binary masking to remove background
        3. Applies Gaussian smoothing (variance = 0.015)
        4. Resamples to target resolution using linear interpolation
        5. Maintains original origin and direction

        The output size is calculated based on the ratio of input to output spacing.
        """
        # Need single channel image first because the gaussian filter
        # doesn't support vector images
        single_channel_image = self.get_single_channel_image(channel=channel)
        # Mask the image to remove the background
        binary_mask = self.get_binary_mask(channel=channel)
        single_channel_image = sitk.Mask(single_channel_image, binary_mask)

        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        # XXX: hard coded variance
        gaussian_filter.SetVariance([0.015, 0.015, 0.015])
        smoothed_image = gaussian_filter.Execute(single_channel_image)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(spacing)

        # Calculate output size based on input size and spacing
        input_size = self.size
        input_spacing = self.spacing
        # Calculate output size: new_size = old_size * (old_spacing / new_spacing)
        output_size = [
            int(input_size[0] * input_spacing[0] / spacing[0]),
            int(input_size[1] * input_spacing[1] / spacing[1]),
            int(input_size[2] * input_spacing[2] / spacing[2])
        ]
        resampler.SetSize(output_size)
        resampler.SetOutputOrigin(self.origin)
        resampler.SetOutputDirection(self.direction)
        resampler.SetInterpolator(sitk.sitkLinear)

        resampled_image = resampler.Execute(smoothed_image)

        if save_path is not None:
            sitk.WriteImage(resampled_image, save_path)

        if return_img:
            return resampled_image
        else:
            return None

        return resampled_image


    def c3d_resample_to_mri_resolution(self, smoothing='0.15mm', spacing='0.3mm', save_path=None, return_img=True):
        """
        Resample histology image to MRI resolution using Convert3D (c3d) tools.

        Parameters
        ----------
        smoothing : str, optional
            Smoothing parameter for c3d (e.g., '0.15mm'). Default is '0.15mm'.
        spacing : str, optional
            Target spacing for c3d (e.g., '0.3mm'). Default is '0.3mm'.
        save_path : str, optional
            Path to save the resampled image. If None, image is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.

        Returns
        -------
        SimpleITK.Image or None
            The resampled image if return_img=True, otherwise None.

        Notes
        -----
        This is an alternative to resample_to_mri_resolution() using c3d tools.
        The c3d approach uses fast smoothing and resampling which may be more
        efficient for large images.
        """
        single_channel_image = self.c3d_get_single_channel_image()
        c3d.push(single_channel_image)
        c3d.execute(f'-smooth-fast {smoothing} -resample-mm {spacing}')

        resampled_image = c3d.pop()

        if save_path is not None:
            sitk.WriteImage(resampled_image, save_path)

        if return_img:
            return resampled_image
        else:
            return None


    def get_binary_mask(self, channel=1, save_path=None, return_img=True):
        """
        Create a binary mask from histology image using Otsu thresholding.

        Parameters
        ----------
        channel : int, optional
            Channel index to use for mask creation. Default is 1.
        save_path : str, optional
            Path to save the binary mask. If None, mask is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.

        Returns
        -------
        SimpleITK.Image or None
            The binary mask if return_img=True, otherwise None.

        Notes
        -----
        This function creates a binary mask by:
        1. Extracting the specified channel from multi-channel input
        2. Applying Otsu thresholding to separate tissue from background
        3. Performing morphological closing to fill holes in the mask
        4. Using a ball-shaped structuring element with radius [2, 2, 2]

        The resulting mask can be used for weighted registration to focus
        on tissue regions and ignore background areas.
        """
        # get the single channel image
        single_channel = self.get_single_channel_image(channel=channel)

        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(1)
        otsu_filter.SetOutsideValue(0)
        binary_mask = otsu_filter.Execute(single_channel)

        # Otsu thresholding has holes in the mask due to the irregular staining
        # Dilation to fill the holes
        structuring_element = sitk.sitkBall
        # XXX: hard coded radius
        radius = [2, 2, 2]
        closed_mask = sitk.BinaryMorphologicalClosing(binary_mask, radius, structuring_element)

        if save_path is not None:
            sitk.WriteImage(closed_mask, save_path)

        if return_img:
            return closed_mask
        else:
            return None


    def c3d_get_lowres_mask(self, binary_mask, resampled_histo, save_path=None, return_img=True):
        """
        Create a low-resolution mask by resampling a binary mask.

        Parameters
        ----------
        binary_mask : SimpleITK.Image
            High-resolution binary mask to be resampled.
        resampled_histo : SimpleITK.Image
            Reference image whose size will be used for resampling.
        save_path : str, optional
            Path to save the low-resolution mask. If None, mask is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.

        Returns
        -------
        SimpleITK.Image or None
            The low-resolution mask if return_img=True, otherwise None.

        Notes
        -----
        This function resamples a high-resolution binary mask to match the size
        of a resampled histology image. The resampling is done using c3d tools
        to ensure consistency with the histology processing pipeline.

        The low-resolution mask is used during registration to provide spatial
        weighting and focus the alignment on tissue regions.
        """
        output_size = resampled_histo.GetSize()

        # binary_mask_np = sitk.GetArrayFromImage(binary_mask)
        # binary_mask = sitk.GetImageFromArray(binary_mask_np, isVector=False)

        c3d.push(binary_mask)
        c3d.execute(f'-resample {output_size[0]}x{output_size[1]}x{output_size[2]}')

        lowres_mask = c3d.pop()

        if save_path is not None:
            sitk.WriteImage(lowres_mask, save_path)

        if return_img:
            return lowres_mask
        else:
            return None


    def preprocess_histology(self, channel=1, base_dir=None):
        """
        Complete preprocessing pipeline for histology data.

        Parameters
        ----------
        channel : int, optional
            Channel index to process from multi-channel input. Default is 1.
        base_dir : str
            Base directory where all processed files will be saved.
            Must be provided.

        Raises
        ------
        ValueError
            If base_dir is None.

        Notes
        -----
        This function implements the complete histology preprocessing pipeline:

        1. **Single Channel Extraction**: Extracts the specified channel from
           multi-channel histology and saves as 'historef_single_channel.nii.gz'

        2. **Resampling**: Resamples the histology to MRI resolution with
           smoothing and saves as 'historef_resampled.nii.gz'

        3. **Binary Mask Creation**: Creates a binary mask using Otsu thresholding
           and morphological closing, saves as 'historef_binary_mask.nii.gz'

        4. **Low-Resolution Mask**: Resamples the binary mask to match the
           resampled histology size, saves as 'historef_lowres_mask.nii.gz'

        All files are saved in the specified base directory and are ready for
        use in the registration pipeline.
        """
        if base_dir is None:
            raise ValueError("Base directory is required to save images")

        # Step 1 - Extract single channel histology
        historef_single_channel_path = os.path.join(base_dir, "historef_single_channel.nii.gz")
        histo_single_channel = self.get_single_channel_image(channel=channel, save_path=historef_single_channel_path)

        # Step 2 - Resample histology to MRI resolution
        historef_resampled_path = os.path.join(base_dir, "historef_resampled.nii.gz")
        histo_resampled = self.resample_to_mri_resolution(channel=channel, save_path=historef_resampled_path)

        # Step 3 - Get binary mask
        historef_binary_mask_path = os.path.join(base_dir, "historef_binary_mask.nii.gz")
        histo_binary = self.get_binary_mask(channel=channel, save_path=historef_binary_mask_path)

        # Step 4 - Get lowres mask
        historef_lowres_mask_path = os.path.join(base_dir, "historef_lowres_mask.nii.gz")
        histo_lowres = self.c3d_get_lowres_mask(histo_binary, histo_resampled, save_path=historef_lowres_mask_path)

        print(f"Reference histology slide preprocessed and saved to {base_dir}")
