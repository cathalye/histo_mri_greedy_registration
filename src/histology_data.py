import subprocess

import numpy as np
import SimpleITK as sitk

from picsl_c3d import Convert3D
from picsl_image_graph_cut import image_graph_cut

c3d = Convert3D()

class HistologyData:

    def __init__(self, histology_path):
        self.histology_path = histology_path
        self.sitk_image = sitk.ReadImage(histology_path, sitk.sitkVectorFloat32)

        # Extract metadata
        self.spacing = self.sitk_image.GetSpacing()
        self.size = self.sitk_image.GetSize()
        self.origin = self.sitk_image.GetOrigin()
        self.direction = self.sitk_image.GetDirection()


    def get_single_channel_image(self, channel=0):
        single_channel_image = sitk.VectorIndexSelectionCast(self.sitk_image, channel)
        return single_channel_image


    def c3d_get_single_channel_image(self, channel=0):
        # TODO: ask Paul why -mcs throws an error
        # Also had the same issue with data_histo_mri_paired_png
        # used subprocess to get around it there
        c3d.push(self.sitk_image)
        c3d.execute(f'-mcs -pick {channel}')

        return c3d.pop()


    def resample_to_mri_resolution(self, spacing=[0.3, 0.3, 1.0]):
        # Need single channel image first because the gaussian filter
        # doesn't support vector images
        single_channel_image = self.get_single_channel_image()

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

        return resampled_image


    def c3d_resample_to_mri_resolution(self, smoothing='0.15mm', spacing='0.3mm'):
        single_channel_image = self.c3d_get_single_channel_image()
        c3d.push(single_channel_image)
        c3d.execute(f'-smooth-fast {smoothing} -resample-mm {spacing}')

        return c3d.pop()


    def _remove_mask_border(self, mask, border=25):
        # A lot of slides have shadow/dark artefacts on the border that affect
        # the mask thresholding. We remove the border of the mask to avoid this.
        mask_arr = sitk.GetArrayFromImage(mask)[0, :, :]
        mask_arr[:border, :] = 0
        mask_arr[-border:, :] = 0
        mask_arr[:, :border] = 0
        mask_arr[:, -border:] = 0

        return sitk.GetImageFromArray(mask_arr)


    def get_binary_mask(self, channel=1):
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

        closed_mask = self._remove_mask_border(closed_mask)

        return closed_mask


    def get_lowres_mask(self, binary_mask, resampled_histo):
        output_size = resampled_histo.GetSize()

        # Do this theatre to get a 3D mask
        # because c3d doesn't support 2D masks
        # and simpleitk squeezes the singleton dimension
        # OR to avoid all this,accept the mask path and have c3d read it
        binary_mask_np = sitk.GetArrayFromImage(binary_mask)
        binary_mask_3d = binary_mask_np[np.newaxis, :, :]
        binary_mask = sitk.GetImageFromArray(binary_mask_3d, isVector=False)

        c3d.push(binary_mask)
        c3d.execute(f'-resample {output_size[0]}x{output_size[1]}x{output_size[2]}')

        return c3d.pop()


    def get_chunk_mask(self, binary_mask_path, chunk_mask_path, n_parts=10):
        # BUG: There are no registered IO factories
        # RuntimeError: /Users/runner/work/image-graph-cut/image-graph-cut/be/install/include/ITK-5.4/itkImageFileReader.hxx:135:
        #  Could not create IO object for reading file /Users/cathalye/Projects/proj_histo_mri_greedy_registration/scratch/binary.nii.gz
        #  There are no registered IO factories.
        #  Please visit https://www.itk.org/Wiki/ITK/FAQ#NoFactoryException to diagnose the problem.

        # XXX: hard coded parameters
        # image_graph_cut(
        #     fn_input=binary_mask_path,
        #     fn_output=chunk_mask_path,
        #     n_parts=n_parts,
        #     tolerance=1.2,
        #     n_metis_iter=100,
        #     max_comp=4,
        #     min_comp_frac=0.1
        # )

        subprocess.run([
            "image_graph_cut",
            "-u", "1.2",
            "-n", "100",
            "-c", "4", "0.1",
            binary_mask_path,
            chunk_mask_path,
            str(n_parts),
        ], stdout=subprocess.DEVNULL)
