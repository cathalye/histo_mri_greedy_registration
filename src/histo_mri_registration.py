import os
import subprocess

import numpy as np
import SimpleITK as sitk
from picsl_greedy import Greedy3D

greedy = Greedy3D()


class HistoMRIRegistration:

    def __init__(self, base_dir):
        # Paths to all files for registration
        self.base_dir = base_dir
        self.mri_slab_path = os.path.join(base_dir, f"mri_slab.nii.gz")
        self.histo_resampled_path = os.path.join(base_dir, "histology/historef_resampled.nii.gz")
        self.histo_lowres_mask_path = os.path.join(base_dir, "histology/historef_lowres_mask.nii.gz")

        # Transforms
        self.centers_init_transform_path = os.path.join(base_dir, "transforms/centers_init.mat")
        self.manual_init_workspace_path = os.path.join(base_dir, "manual/manual_init.itksnap")
        self.manual_result_workspace_path = os.path.join(base_dir, "manual/manual_init_result.itksnap")
        self.manual_init_transform_path = os.path.join(base_dir, "transforms/manual_init.mat")
        self.rigid_transform_path = os.path.join(base_dir, "transforms/rigid.mat")
        self.affine_transform_path = os.path.join(base_dir, "transforms/affine.mat")
        self.deformable_transform_path = os.path.join(base_dir, "transforms/deformable.mhd")

        # Resliced images for each step
        self.manual_result_path = os.path.join(base_dir, "manual_result.nii.gz")
        self.rigid_result_path = os.path.join(base_dir, "rigid_result.nii.gz")
        self.affine_result_path = os.path.join(base_dir, "affine_result.nii.gz")
        self.deformable_result_path = os.path.join(base_dir, "deformable_result.nii.gz")

        # ITK-SNAP workspace
        self.results_workspace_path = os.path.join(base_dir, "results.itksnap")


    def _get_physical_center_of_image(self, image, c3d=True):
        size = image.GetSize()

        center = np.zeros(len(size))
        for i in range(len(size)):
            center[i] = size[i] / 2

        if c3d:
            # when computing physical point coordinates, the signs of x, y
            # coordinate are flipped between sitk (LPS) and c3d (RAS) conventions!
            phys_center = image.TransformContinuousIndexToPhysicalPoint(center)
            return (-phys_center[0], -phys_center[1], phys_center[2])
        else:
            return image.TransformContinuousIndexToPhysicalPoint(center)


    def create_initial_alignment(self):
        # Extract MRI slab
        mri_slab = sitk.ReadImage(self.mri_slab_path)
        histo_resampled = sitk.ReadImage(self.histo_resampled_path)

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

        mri_center = self._get_physical_center_of_image(mri_slab, c3d=True)
        mri_center = ', '.join(map(str, mri_center))
        histology_center = self._get_physical_center_of_image(histo_resampled, c3d=True)
        histology_center = ', '.join(map(str, histology_center))

        subprocess.run(
            f"c3d_affine_tool \
                -tran {mri_center} \
                -rot 90 1 0 0 \
                -tran {histology_center} \
                -inv -mult -mult -info \
                -o {self.centers_init_transform_path}",
            shell=True, stdout=subprocess.DEVNULL
        )

    def save_manual_itksnap_workspace(self):
        subprocess.run([
            "itksnap-wt",
                "-layers-add-anat", self.histo_resampled_path, "-psn", "Fixed image",
                "-layers-add-anat", self.mri_slab_path, "-psn", "Moving image", "-props-set-transform", self.centers_init_transform_path,
                "-layers-add-seg", self.histo_lowres_mask_path,
                "-o", self.manual_init_workspace_path
                ], stdout=subprocess.DEVNULL
           )

    # Extract the manual alignment matrix from initial alignment workspace
    def extract_manual_alignment_matrix(self):
        subprocess.run(
            f"itksnap-wt -i {self.manual_result_workspace_path} -lp 1 -props-get-transform | grep '3>' | sed -e 's/3> //g' > {self.manual_init_transform_path}",
            shell=True, stdout=subprocess.DEVNULL
            )


    def rigid_registration(self):
        greedy.execute('-d 3 '
                       '-z '
                       '-a -dof 7 '
                       '-i {} {} '
                       '-gm {} '
                       '-ia {} '
                       '-m WNCC 2x2x0 '
                       '-bg NaN '
                       '-n 100x100x80 '
                       '-o {} '.format(self.histo_resampled_path, self.mri_slab_path,
                                       self.histo_lowres_mask_path,
                                       self.manual_init_transform_path,
                                       self.rigid_transform_path)
                       )

    def affine_registration(self):
        greedy.execute('-d 3 '
                       '-z '
                       '-a -dof 12 '
                       '-i {} {} '
                       '-gm {} '
                       '-ia {} '
                       '-m WNCC 2x2x0 '
                       '-bg NaN '
                       '-n 100x100x40 '
                       '-o {} '.format(self.histo_resampled_path, self.mri_slab_path,
                                       self.histo_lowres_mask_path,
                                       self.rigid_transform_path,
                                       self.affine_transform_path)
                       )

    def deformable_registration(self):
        greedy.execute('-d 3 '
                       '-z '
                       '-ref-pad 0x0x2 '
                       '-i {} {} '
                       '-gm {} '
                       '-it {} '
                       '-m WNCC 2x2x0 '
                       '-bg NaN '
                       '-n 200x200x100 '
                       '-s 1.5mm 0.5mm '
                       '-sv '
                       '-e 0.5 '
                       '-o {} '.format(self.histo_resampled_path, self.mri_slab_path,
                                       self.histo_lowres_mask_path,
                                       self.affine_transform_path,
                                       self.deformable_transform_path)
                       )

    def apply_transforms(self):
        # Result after applying all the transforms to the MRI slab
        greedy.execute('-d 3 '
                       '-rf {} '
                       '-rm {} {} '
                       '-r {} {} '.format(self.histo_resampled_path,
                                          self.mri_slab_path, self.deformable_result_path,
                                          self.deformable_transform_path, self.affine_transform_path)
                       )

        # Result after applying the affine transform
        greedy.execute('-d 3 '
                       '-rf {} '
                       '-rm {} {} '
                       '-r {} '.format(self.histo_resampled_path,
                                       self.mri_slab_path, self.affine_result_path,
                                       self.affine_transform_path)
                       )

        # Result after applying the rigid transform
        greedy.execute('-d 3 '
                       '-rf {} '
                       '-rm {} {} '
                       '-r {} '.format(self.histo_resampled_path,
                                       self.mri_slab_path, self.rigid_result_path,
                                       self.rigid_transform_path)
                       )

        # Result after applying the manual transform
        greedy.execute('-d 3 '
                       '-rf {} '
                       '-rm {} {} '
                       '-r {} '.format(self.histo_resampled_path,
                                       self.mri_slab_path, self.manual_result_path,
                                       self.manual_init_transform_path)
                       )

        subprocess.run([
            "itksnap-wt",
                "-layers-add-anat", self.histo_resampled_path, "-psn", "Fixed image",
                "-layers-add-anat", self.deformable_result_path, "-psn", "MRI (deformable)",
                "-layers-add-anat", self.affine_result_path, "-psn", "MRI (affine)",
                "-layers-add-anat", self.rigid_result_path, "-psn", "MRI (rigid)",
                "-layers-add-anat", self.manual_result_path, "-psn", "MRI (manual)",
                "-layers-add-seg", self.histo_lowres_mask_path, "-psn", "Mask",
                "-o", self.results_workspace_path
        ], stdout=subprocess.DEVNULL
        )
