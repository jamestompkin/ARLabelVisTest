import numpy as np
from scipy.spatial import KDTree, ConvexHull
import math
from arlabelvis.colors import sRGBtoLAB
from skimage.color import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, lab2rgb
import pyvista as pv

def closest_vertices_batch(lab_points: np.ndarray, mesh_tree: KDTree) -> np.ndarray:
    """For each LAB point, return the index of the closest mesh vertex."""
    _, indices = mesh_tree.query(lab_points)
    return indices.astype(int)

def closest_labs_batch(mesh_vertices: np.ndarray, lab_tree: KDTree) -> np.ndarray:
    """For each mesh vertex, return the index of the closest LAB point."""
    _, indices = lab_tree.query(mesh_vertices)
    return indices.astype(int)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def furthest_rgd(vertices, allPoints, allRGBs, matlab_path):
    allPoints = np.asarray(allPoints, dtype=float)
    allRGBs = np.asarray(allRGBs)

    mesh_tree = KDTree(vertices)
    lab_tree = KDTree(allPoints)
    LABtoVertices = closest_vertices_batch(allPoints, mesh_tree)
    VerticestoLAB = closest_labs_batch(vertices, lab_tree)
    print(f"{len(allPoints)} LAB points <-> {len(vertices)} mesh vertices mapped")

    unique_sources = np.unique(LABtoVertices)
    n_unique = len(unique_sources)

    print(f"Finding furthest for {n_unique} unique sources ({len(allPoints)} LAB points)")

    furthestRGBs = []
    max_indices = np.loadtxt(matlab_path, delimiter=',', dtype='int')
    print(f"From matlab file: {matlab_path}")

    for i in range(allPoints.shape[0]):
        vert = LABtoVertices[i]
        if vert < max_indices.shape[0]:
            furthest_vert = max_indices[vert] - 1
        else:
            furthest_vert = 0
            print("WARNING: out of bounds. Vertices are likely not what RGD was run on.")
        furthest_lab = VerticestoLAB[int(furthest_vert)]
        furthestRGBs.append(allRGBs[int(furthest_lab)])

    return np.array(furthestRGBs)

def furthest_delta_e76_points(inputRGB, allLABPoints):
    # Note: ΔE₇₆ is by definition sqrt((ΔL)² + (Δa)² + (Δb)²) — i.e. Euclidean
    # distance in CIELAB. This path is kept because it returns hull-LAB values
    # (interpolated in LAB by downstream stages) rather than grid sRGB values
    # (interpolated in sRGB by `furthest_euclidean_lab_points`).
    inputLAB = sRGBtoLAB([inputRGB])[0]
    allLABPoints = np.array(allLABPoints)

    distances = deltaE_cie76(np.tile(inputLAB, (len(allLABPoints), 1)), allLABPoints)
    max_distance_index = np.argmax(distances)

    return allLABPoints[max_distance_index]


def furthest_delta_e94_points(inputRGB, allLABPoints):
    """Farthest CIELAB point from inputRGB under ΔE₉₄ (CIE94, graphic-arts
    weights). Returns the chosen LAB coordinate."""
    inputLAB = sRGBtoLAB([inputRGB])[0]
    allLABPoints = np.asarray(allLABPoints)
    tiled = np.tile(inputLAB, (len(allLABPoints), 1))
    distances = deltaE_ciede94(tiled, allLABPoints)
    return allLABPoints[int(np.argmax(distances))]


def furthest_delta_e00_points(inputRGB, allLABPoints):
    """Farthest CIELAB point from inputRGB under ΔE₀₀ (CIEDE2000). Returns
    the chosen LAB coordinate."""
    inputLAB = sRGBtoLAB([inputRGB])[0]
    allLABPoints = np.asarray(allLABPoints)
    tiled = np.tile(inputLAB, (len(allLABPoints), 1))
    distances = deltaE_ciede2000(tiled, allLABPoints)
    return allLABPoints[int(np.argmax(distances))]

def furthest_euclidean_lab_points(allLABs, chunk_size=10_000):
    lab = allLABs.astype(np.float32)

    hull = ConvexHull(lab)
    hull_verts = lab[hull.vertices].astype(np.float32)
    print(f"Hull vertices: {len(hull_verts)}")

    N = len(lab)
    furthest = np.empty((N, 3), dtype=np.float32)
    for i in range(0, N, chunk_size):
        print(i)
        chunk = lab[i:i+chunk_size]
        dists_sq = np.sum((chunk[:, None, :] - hull_verts[None, :, :]) ** 2, axis=-1)
        furthest[i:i+chunk_size] = 255.0 * lab2rgb(hull_verts[np.argmax(dists_sq, axis=1)])
    return furthest

def furthest_euclidean_rgb(allRGBs):
    furthest = np.where(allRGBs < 128, 255, 0).astype(np.uint8) 
    return furthest
