import os
import subprocess

import SimpleITK as sitk

from mri_data import MRIData
from histology_data import HistologyData
import utils


class HistoMRIRegistration:

    def __init__(self, mri_data: MRIData, histology_data: HistologyData, slab_num: int):
        self.mri_data = mri_data
        self.histology_data = histology_data
        self.slab_num = slab_num

    def save_manual_itksnap_workspace(self, image1_path, image2_path, workspace_path,
                                      segmentation=None, transform=None,
                                      image1_description="Fixed image",
                                      image2_description="Moving image"):
        # TODO: make this modular to work with N images, image descriptions, and transforms

        # TODO: Ask Paul does itksnap-wt have python wrapper?

        subprocess.run([
            "itksnap-wt",
                "-layers-add-anat", image1_path, "-psn", image1_description,
                "-layers-add-anat", image2_path, "-psn", image2_description, "-props-set-transform", transform,
                "-layers-add-seg", segmentation,
                "-o", workspace_path
        ],stdout=subprocess.DEVNULL
        )

        return None

    def create_initial_alignment(self, output_path):
        # Extract MRI slab
        mri_slab = self.mri_data.get_mri_slab(self.slab_num)

        # TODO: Implement initial alignment logic
        # Convert c3d_affine tool command to simpleitk commands
        # -probe 50% gets the center of the image
        # cross check the output of c3d -probe 50% with simpleitk implementation
        # generate a translation matric using this center
        # sitk.TranslationTransform(n_dims, center)
        # confirm the output of this with c3d_affine_tool -tran vx vy vz
        # do this for MRI slab and histology
        # get the inverse and multiply them together
        # save the result as init.mat
        # cross check the output of this with c3d_affine_tool output
        # init.mat will be applied to the mri_slab for center alignment
        # - Creating ITK-SNAP workspace

        mri_center = utils.get_physical_center_of_image(mri_slab, c3d=True)
        mri_center = ', '.join(map(str, mri_center))
        histology_center = utils.get_physical_center_of_image(self.histology_data.sitk_image, c3d=True)
        histology_center = ', '.join(map(str, histology_center))

        subprocess.run(
            f"c3d_affine_tool \
                -tran {mri_center} \
                -rot 90 1 0 0 \
                -tran {histology_center} \
                -inv -mult -mult -info \
                -o {output_path}",
            shell=True, stdout=subprocess.DEVNULL
        )

        return None
