import os

import numpy as np
import SimpleITK as sitk
from skimage import color, filters, morphology

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

    def _load_existing_if_needed(self, save_path, overwrite, return_img):
        """
        Helper method to check if existing file should be loaded instead of recomputing.

        Parameters
        ----------
        save_path : str or None
            Path to the file to check.
        overwrite : bool
            Whether to overwrite existing files.
        return_img : bool
            Whether to return the image object.

        Returns
        -------
        SimpleITK.Image or None or False
            Returns the loaded image if file exists and should be loaded,
            False if computation should proceed, None if file exists but return_img=False.
        """
        if save_path is not None and not overwrite and os.path.exists(save_path):
            return sitk.ReadImage(save_path) if return_img else None
        return False

    def get_single_channel_image(self, channel=1, save_path=None, return_img=True, overwrite=False):
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
        overwrite : bool, optional
            If True, recompute and overwrite existing files. If False, skip computation
            if file already exists. Default is False.

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
        existing = self._load_existing_if_needed(save_path, overwrite, return_img)
        if existing is not False:
            return existing

        single_channel_image = sitk.VectorIndexSelectionCast(self.sitk_image, channel)

        if save_path is not None:
            sitk.WriteImage(single_channel_image, save_path)

        if return_img:
            return single_channel_image
        return None


    def c3d_get_single_channel_image(self, channel=1, save_path=None, return_img=True, overwrite=False):
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
        overwrite : bool, optional
            If True, recompute and overwrite existing files. If False, skip computation
            if file already exists. Default is False.

        Returns
        -------
        SimpleITK.Image or None
            The extracted single-channel image if return_img=True, otherwise None.

        Notes
        -----
        This is an alternative to get_single_channel_image() using c3d tools.
        """
        existing = self._load_existing_if_needed(save_path, overwrite, return_img)
        if existing is not False:
            return existing

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
        return None


    def resample_to_mri_resolution(self, channel=1, spacing=[0.3, 0.3, 1.0], save_path=None, return_img=True, overwrite=False):
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
        overwrite : bool, optional
            If True, recompute and overwrite existing files. If False, skip computation
            if file already exists. Default is False.

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
        existing = self._load_existing_if_needed(save_path, overwrite, return_img)
        if existing is not False:
            return existing

        # Need single channel image first because the gaussian filter
        # doesn't support vector images
        single_channel_image = self.get_single_channel_image(channel=channel, overwrite=overwrite)
        # Mask the image to remove the background
        binary_mask = self.get_binary_mask(overwrite=overwrite)
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
        return None


    def c3d_resample_to_mri_resolution(self, smoothing='0.15mm', spacing='0.3mm', save_path=None, return_img=True, overwrite=False):
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
        overwrite : bool, optional
            If True, recompute and overwrite existing files. If False, skip computation
            if file already exists. Default is False.

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
        existing = self._load_existing_if_needed(save_path, overwrite, return_img)
        if existing is not False:
            return existing

        single_channel_image = self.c3d_get_single_channel_image(overwrite=overwrite)
        c3d.push(single_channel_image)
        c3d.execute(f'-smooth-fast {smoothing} -resample-mm {spacing}')

        resampled_image = c3d.pop()

        if save_path is not None:
            sitk.WriteImage(resampled_image, save_path)

        if return_img:
            return resampled_image
        return None


    def get_binary_mask(self, save_path=None, return_img=True, overwrite=False):
        """
        Create a binary mask from histology image using saturation thresholding.

        Parameters
        ----------
        save_path : str, optional
            Path to save the binary mask. If None, mask is not saved.
        return_img : bool, optional
            Whether to return the SimpleITK image object. Default is True.
        overwrite : bool, optional
            If True, recompute and overwrite existing files. If False, skip computation
            if file already exists. Default is False.

        Returns
        -------
        SimpleITK.Image or None
            The binary mask if return_img=True, otherwise None.
        """
        existing = self._load_existing_if_needed(save_path, overwrite, return_img)
        if existing is not False:
            return existing

        rgb_array = sitk.GetArrayFromImage(self.sitk_image)

        # Handle 4D arrays from NIfTI (shape: 1, H, W, 3) -> squeeze to (H, W, 3)
        original_shape = rgb_array.shape
        if rgb_array.ndim == 4 and rgb_array.shape[0] == 1:
            rgb_array = rgb_array.squeeze(axis=0)

        if rgb_array.max() > 1:
            rgb_normalized = rgb_array.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_array.astype(np.float32)

        # Convert RGB to HSV and extract saturation channel
        hsv = color.rgb2hsv(rgb_normalized)
        saturation = hsv[:, :, 1]

        # Triangle threshold: optimal for unimodal + tail distributions
        # (large background peak at low saturation, tissue spread higher)
        thresh = filters.threshold_triangle(saturation)
        thresh = max(thresh, 0.015)  # Floor to avoid noise sensitivity

        # Binary mask: tissue has color, background doesn't
        tissue_mask = saturation > thresh

        # Morphological cleanup
        mask = morphology.remove_small_objects(tissue_mask, min_size=100)
        mask = morphology.remove_small_holes(mask, area_threshold=500)
        mask = morphology.binary_closing(mask, morphology.disk(2)).astype(np.uint8)

        if len(original_shape) == 4 and original_shape[0] == 1:
            mask_out = mask[np.newaxis, ...]
        else:
            mask_out = mask

        mask_image = sitk.GetImageFromArray(mask_out)
        mask_image.CopyInformation(self.sitk_image)

        if save_path is not None:
            sitk.WriteImage(mask_image, save_path)

        if return_img:
            return mask_image
        return None


    def c3d_get_lowres_mask(self, binary_mask, resampled_histo, save_path=None, return_img=True, overwrite=False):
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
        overwrite : bool, optional
            If True, recompute and overwrite existing files. If False, skip computation
            if file already exists. Default is False.

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
        existing = self._load_existing_if_needed(save_path, overwrite, return_img)
        if existing is not False:
            return existing

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
        return None


    def preprocess_histology(self, channel=1, base_dir=None, overwrite=False):
        """
        Complete preprocessing pipeline for histology data.

        Parameters
        ----------
        channel : int, optional
            Channel index to process from multi-channel input. Default is 1.
        base_dir : str
            Base directory where all processed files will be saved.
            Must be provided.
        overwrite : bool, optional
            If True, recompute and overwrite existing files. If False, skip computation
            if file already exists. Default is False.

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
        histo_single_channel = self.get_single_channel_image(channel=channel, save_path=historef_single_channel_path, overwrite=overwrite)

        # Step 2 - Resample histology to MRI resolution
        historef_resampled_path = os.path.join(base_dir, "historef_resampled.nii.gz")
        histo_resampled = self.resample_to_mri_resolution(channel=channel, save_path=historef_resampled_path, overwrite=overwrite)

        # Step 3 - Get binary mask
        historef_binary_mask_path = os.path.join(base_dir, "historef_binary_mask.nii.gz")
        histo_binary = self.get_binary_mask(save_path=historef_binary_mask_path, overwrite=overwrite)

        # Step 4 - Get lowres mask
        historef_lowres_mask_path = os.path.join(base_dir, "historef_lowres_mask.nii.gz")
        histo_lowres = self.c3d_get_lowres_mask(histo_binary, histo_resampled, save_path=historef_lowres_mask_path, overwrite=overwrite)

        print(f"Reference histology slide preprocessed and saved to {base_dir}")
