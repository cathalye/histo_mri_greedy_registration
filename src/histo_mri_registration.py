import os
import subprocess

import numpy as np
import SimpleITK as sitk
from picsl_greedy import Greedy3D

greedy = Greedy3D()


class HistoMRIRegistration:
    """
    A class for performing multi-stage registration between postmortem MRI and histology data.

    This class implements a complete registration pipeline that includes:
    1. Initial center-based alignment
    2. Manual refinement using ITK-SNAP
    3. Rigid registration (6-DOF)
    4. Affine registration (12-DOF)
    5. Deformable registration (non-linear)
    6. Transform application and result generation

    The registration uses the Greedy algorithm with Weighted Normalized Cross-Correlation (WNCC)
    as the similarity metric.
    """

    def __init__(self, base_dir):
        """
        Initialize the registration processor with file paths.

        Parameters
        ----------
        base_dir : str
            Base directory containing all input and output files for the registration.
            This directory should contain the MRI slab and histology data.

        Notes
        -----
        Sets up all file paths for:
        - Input images (MRI slab, resampled histology, histology mask)
        - Transform files (initial, manual, rigid, affine, deformable)
        - Result images (registered MRI at each stage)
        - ITK-SNAP workspaces (manual alignment, final results)
        """
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
        """
        Calculate the physical center coordinates of an image.

        Parameters
        ----------
        image : SimpleITK.Image
            The input image for which to calculate the center.
        c3d : bool, optional
            If True, converts coordinates from SimpleITK (LPS) to c3d (RAS) convention.
            Default is True.

        Returns
        -------
        tuple
            Physical center coordinates (x, y, z) in millimeters.
            If c3d=True, coordinates are in RAS convention; otherwise in LPS.

        Notes
        -----
        SimpleITK uses LPS (Left-Posterior-Superior) coordinate system, while c3d uses
        RAS (Right-Anterior-Superior). When c3d=True, the x and y coordinates are flipped
        to convert between these conventions.
        """
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
        """
        Create initial center-based alignment between MRI and histology images.

        This function performs a rough alignment by:
        1. Calculating the physical centers of both MRI and histology images
        2. Creating a transformation that aligns these centers
        3. Applying a 90-degree rotation around the x-axis
        4. Saving the initial transformation matrix

        The alignment uses c3d_affine_tool to create a transformation that:
        - Translates the MRI center to origin
        - Applies a 90-degree rotation around x-axis
        - Translates to the histology center
        - Inverts and multiplies transformations to get the final alignment

        Notes
        -----
        This is a coarse alignment that should be refined manually in ITK-SNAP.
        The transformation is saved as 'centers_init.mat' in the transforms directory.
        """
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
        """
        Create an ITK-SNAP workspace for manual alignment refinement.

        This function generates an ITK-SNAP workspace file that contains:
        - Fixed image: Resampled histology (reference)
        - Moving image: MRI slab with initial center alignment applied
        - Segmentation: Histology mask for guidance

        The workspace is saved as 'manual_init.itksnap' and can be opened in ITK-SNAP
        for manual refinement of the initial alignment.

        Notes
        -----
        The workspace uses the initial center-based transformation as a starting point.
        Users should manually align the images in ITK-SNAP and save the result as
        'manual_init_result.itksnap'.
        """
        subprocess.run([
            "itksnap-wt",
                "-layers-add-anat", self.histo_resampled_path, "-psn", "Fixed image",
                "-layers-add-anat", self.mri_slab_path, "-psn", "Moving image", "-props-set-transform", self.centers_init_transform_path,
                "-layers-add-seg", self.histo_lowres_mask_path,
                "-o", self.manual_init_workspace_path
                ], stdout=subprocess.DEVNULL
           )

    def extract_manual_alignment_matrix(self):
        """
        Extract the manual alignment transformation matrix from ITK-SNAP workspace.

        This function reads the manually refined transformation from the ITK-SNAP
        workspace file and extracts it as a transformation matrix. The extraction
        uses itksnap-wt command-line tool to get the transform from layer 1
        (the moving image layer) and saves it as a matrix file.

        Notes
        -----
        This function assumes that manual alignment has been completed and saved
        as 'manual_init_result.itksnap'. The extracted transformation will be used
        as the initial transform for the automated registration steps.
        """
        subprocess.run(
            f"itksnap-wt -i {self.manual_result_workspace_path} -lp 1 -props-get-transform | grep '3>' | sed -e 's/3> //g' > {self.manual_init_transform_path}",
            shell=True, stdout=subprocess.DEVNULL
            )


    def rigid_registration(self):
        """
        Perform rigid body registration (6 degrees of freedom).

        This function performs rigid registration using the Greedy algorithm with:
        - 6 degrees of freedom (3 translations + 3 rotations)
        - Weighted Normalized Cross-Correlation (WNCC) similarity metric
        - Histology mask for weighted registration
        - Manual alignment as initial transform
        - Multi-resolution optimization (100x100x80 iterations)

        The registration aligns the MRI slab to the histology image while preserving
        the overall shape and size of the MRI data.

        Notes
        -----
        The rigid transformation is saved as 'rigid.mat' and will be used as the
        initial transform for the subsequent affine registration.
        """
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
        """
        Perform affine registration (12 degrees of freedom).

        This function performs affine registration using the Greedy algorithm with:
        - 12 degrees of freedom (3 translations + 3 rotations + 3 scaling + 3 shearing)
        - Weighted Normalized Cross-Correlation (WNCC) similarity metric
        - Histology mask for weighted registration
        - Rigid registration result as initial transform
        - Multi-resolution optimization (100x100x40 iterations)

        The affine registration allows for scaling and shearing transformations
        to better align the MRI and histology images.

        Notes
        -----
        The affine transformation is saved as 'affine.mat' and will be used as the
        initial transform for the subsequent deformable registration.
        """
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
        """
        Perform deformable (non-linear) registration.

        This function performs deformable registration using the Greedy algorithm with:
        - Non-linear deformation field
        - Weighted Normalized Cross-Correlation (WNCC) similarity metric
        - Histology mask for weighted registration
        - Affine registration result as initial transform
        - Multi-resolution optimization (200x200x100 iterations)
        - Multi-scale smoothing (1.5mm to 0.5mm)
        - Convergence threshold of 0.5

        The deformable registration allows for local non-linear deformations to
        achieve the highest accuracy alignment between MRI and histology.

        Notes
        -----
        The deformable transformation is saved as 'deformable.mhd' and represents
        the final registration result. This step requires the most computational
        time but provides the highest accuracy.
        """
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
        """
        Apply all registration transforms and generate result images.

        This function applies the registration transforms in sequence and generates
        registered MRI images at each stage of the pipeline:
        1. Manual alignment result
        2. Rigid registration result
        3. Affine registration result
        4. Deformable registration result (final)

        Additionally, it creates a comprehensive ITK-SNAP workspace containing all
        results for visualization and comparison.

        Notes
        -----
        Each result image shows the MRI data transformed to align with the histology
        reference. The final workspace includes all stages for easy comparison of
        registration quality at each step.
        """
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
