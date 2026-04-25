"""DEPRECATED — sRGB-grid → CIELAB → binvox writer for the external
neural-bounding trainer pipeline.

The pipeline is retired: ``shape='neural'`` now uses the in-process MLP
in ``arlabelvis.neural_bounding``. Nothing in the live pipeline imports
this module. Known bugs preserved verbatim because no caller exercises
them: ``write_voxels`` ignores its ``filename`` argument and always
writes to ``"all_labs.binvox"`` in the current working directory; the
commented-out block in ``convertToVoxels`` is a duplicate of the
function below; ``binvox_rw.write`` is broken under Python 3.
"""
import numpy as np
from arlabelvis.binvox_rw import write, Voxels
from arlabelvis.colors import srgb_to_lab

# def convertToVoxels(all_points, dim):
#     # print("expected size: " + str(len(all_labs)))
#     x_max, y_max, z_max = np.max(all_points, axis=0)
#     x_min, y_min, z_min = np.min(all_points, axis=0)

#     max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

#     voxels = np.zeros((dim, dim, dim), dtype=bool)

#     for lab in all_points:
#         print("LAB")
#         x = lab[0] - x_min
#         y = lab[1] - y_min
#         z = lab[2] - z_min
#         print(lab)
#         print([x,y,z])
#         # everything over 0,0,0

#         x = dim / max_range * x 
#         y = dim / max_range * y 
#         z = dim / max_range * z 
#         print([x,y,z])
        
#         x = int(x)
#         y = int(y)
#         z = int(z)
#         print([x,y,z])

#         voxels[x][y][z] = True

#     print(np.min(all_points, axis=0))
#     print(np.max(all_points, axis=0))
#     print(voxels.shape)

#     print("sum of numpy array: " + str(np.sum(voxels)))
#     # print("done converting to voxels")
#     return voxels

def convertToVoxels(all_labs, dim):
    # print("expected size: " + str(len(all_labs)))
    x_max, y_max, z_max = np.max(all_labs, axis=0)
    x_min, y_min, z_min = np.min(all_labs, axis=0)



    max_range = 1.1 * max([x_max - x_min, y_max - y_min, z_max - z_min])

    # print(max_range)

    voxels = np.zeros((dim, dim, dim), dtype=np.bool_)

    for lab in all_labs:
        # print("LAB")
        # print(lab)
        x = lab[0] - x_min
        y = lab[1] - y_min
        z = lab[2] - z_min
        # print("XYZ")
        # print([x,y,z])

        x = dim / max_range * x
        y = dim / max_range * y
        z = dim / max_range * z

        # print("XYZ")
        # print([x,y,z])
       
        x = int(x)
        y = int(y)
        z = int(z)

        # print("INDICES")
        # print([x,y,z])

        voxels[x][y][z] = True

    print(np.min(all_labs, axis=0))
    print(np.max(all_labs, axis=0))
    print(voxels.shape)

    print("sum of numpy array: " + str(np.sum(voxels)))
    # print("done converting to voxels")
    return voxels


def write_voxels(all_points, dim, filename):
    # numpy_voxels = convertToVoxels(all_points, dim)
    # voxels = Voxels(numpy_voxels, [dim, dim, dim], [0.0, 0.0, 0.0], 1.0, 'xyz')
    # with open(filename, "w", encoding="latin-1") as fp:
    #     write(voxels, fp)
    step_size = 16
    all_rgb = np.array([[r-1, g-1, b-1] for r in range(0, 257, step_size)
                               for g in range(0, 257, step_size)
                               for b in range(0, 257, step_size)])
    all_rgb = np.where(all_rgb < 0, 0, all_rgb)
    all_rgb = np.where(all_rgb > 255, 255, all_rgb)
    all_labs = srgb_to_lab(all_rgb)

    dim = 32
    voxels = convertToVoxels(all_labs, dim)
    v = Voxels(voxels, [dim, dim, dim], [0.0, 0.0, 0.0], 1.0, 'xyz')
    filepath = "all_labs.binvox"
    with open(filepath, 'w', encoding="latin-1") as fp:
        write(v, fp)
    print("Saved to file " + filepath)
