import nibabel as nib
import SimpleITK as sitk


class MRIData:

    def __init__(self, mri_path):
        self.mri_path = mri_path
        self.sitk_image = sitk.ReadImage(mri_path)
        self.nib_image = nib.load(mri_path)

        # Extract metadata
        self.spacing = self.sitk_image.GetSpacing()
        self.size = self.sitk_image.GetSize()

        # Calculate AP axis information
        self._calculate_ap_axis()


    def _calculate_ap_axis(self):
        affine = self.nib_image.affine
        orientation = nib.orientations.aff2axcodes(affine)

        self.ap_axis = orientation.index('A') if 'A' in orientation else orientation.index('P')
        self.ap_direction = 1 if orientation[self.ap_axis] == 'A' else -1


    def _get_start_end_mm(self, slab_num):
        # Assume each slab is 10 mm thick and each slit in the mold is 2 mm
        start_mm = slab_num * 12 - 4
        end_mm = start_mm + 10
        return start_mm, end_mm


    def _get_slab_start_end_voxels(self, slab_num):
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
