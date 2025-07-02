import numpy as np
import SimpleITK as sitk

def get_physical_center_of_image(image, c3d=True):
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
