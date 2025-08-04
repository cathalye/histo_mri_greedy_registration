import os

import numpy as np
import SimpleITK as sitk
from picsl_c3d import Convert3D

c3d = Convert3D()

class HistologyData:

    def __init__(self, histology_path):
        self.histology_path = histology_path
        self.sitk_image = sitk.ReadImage(histology_path)

        # Extract metadata
        self.spacing = self.sitk_image.GetSpacing()
        self.size = self.sitk_image.GetSize()
        self.origin = self.sitk_image.GetOrigin()
        self.direction = self.sitk_image.GetDirection()


    def get_single_channel_image(self, channel=1, save_path=None, return_img=True):
        single_channel_image = sitk.VectorIndexSelectionCast(self.sitk_image, channel)

        if save_path is not None:
            sitk.WriteImage(single_channel_image, save_path)

        if return_img:
            return single_channel_image
        else:
            return None


    def c3d_get_single_channel_image(self, channel=1, save_path=None, return_img=True):
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
        if base_dir is None:
            raise ValueError("Base directory is required to save images")

        # Step 1 - Extract single channel histology
        historef_single_channel_path = os.path.join(base_dir, "historef_single_channel.nii.gz")
        histo_single_channel = self.get_single_channel_image(channel=channel, save_path=historef_single_channel_path)
        # histo_single_channel = self.c3d_get_single_channel_image(channel=1, save_path=historef_single_channel_path)

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
