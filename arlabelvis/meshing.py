import numpy as np
import trimesh
import trimesh.smoothing
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
import open3d as o3d

def pointsToMesh(allLABs, sigma=0.25, vox=256, pre_decimate_smooth: int = 3, post_decimate_smooth: int = 5, target_faces = 50000):
    print(f"Input length: {len(allLABs):,}")

    mins = allLABs.min(axis=0)
    maxs = allLABs.max(axis=0)
    ranges = maxs - mins + 1e-12

    longest = ranges.max()
    per_axis_res = np.round((ranges / longest) * vox).astype(int)
    per_axis_res = np.clip(per_axis_res, 16, vox)

    padding = 2
    scale = (per_axis_res - 1 - 2 * padding) / ranges

    idx = np.floor((allLABs - mins) * scale).astype(np.int32) + padding
    idx = np.clip(idx, 0, per_axis_res - 1)

    grid = np.zeros(tuple(per_axis_res), dtype=np.float32)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    blurred = gaussian_filter(grid, sigma=sigma)

    isovalue = blurred.max() * 0.5
    voxel_size = ranges / (per_axis_res - 1)

    verts_vox, faces, normals, _ = marching_cubes(blurred, level=isovalue, spacing=tuple(voxel_size))

    verts = verts_vox + mins - (padding * voxel_size)

    tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    if pre_decimate_smooth > 0:
        trimesh.smoothing.filter_laplacian(tm, iterations=pre_decimate_smooth)

    if target_faces and len(tm.faces) > target_faces:
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices  = o3d.utility.Vector3dVector(tm.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        verts = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=tm.vertices.astype(np.float64), face_matrix=tm.faces.astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(iterations=5, targetlen=pymeshlab.PercentageValue(0.8))
    m = ms.current_mesh()
    tm = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=True)

    if post_decimate_smooth > 0:
        trimesh.smoothing.filter_laplacian(tm, iterations=post_decimate_smooth)

    trimesh.repair.fix_normals(tm)
    trimesh.repair.fill_holes(tm)

    components = trimesh.graph.connected_components(tm.edges)
    if len(components) > 1:
        print(f"WARNING: {len(components)} connected components found")
        tm = tm.submesh([max(components, key=len)], append=True)

    print(f"Final mesh: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces, watertight={tm.is_watertight}")
    return tm
    
    
def insideMesh(point: np.array, mesh: trimesh.Trimesh):
    containment = mesh.contains([point])
    return containment[0]


# ---- Mesh optimization (merged from utils/mesh_optimization.py) ----

import numpy as np
import trimesh
import torch
import torch.nn.functional as F
import numpy as np
import trimesh
# torch version: torch-2.5.1 + cu118

def build_sdf_grid(mesh: trimesh.Trimesh, resolution: int = 64):
    bounds_min = mesh.bounds[0].copy()
    bounds_max = mesh.bounds[1].copy()
    padding = (bounds_max - bounds_min) * 0.05
    bounds_min -= padding
    bounds_max += padding

    lin = [np.linspace(bounds_min[i], bounds_max[i], resolution) for i in range(3)]
    xx, yy, zz = np.meshgrid(*lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    chunk_size = 10_000
    distances = np.empty(len(pts), dtype=np.float32)
    signs     = np.empty(len(pts), dtype=np.float32)
    print("len")
    print(len(pts))
    for start in range(0, len(pts), chunk_size):
        end   = min(start + chunk_size, len(pts))
        print(end)
        chunk = pts[start:end]
        _, d, _ = trimesh.proximity.closest_point(mesh, chunk)
        distances[start:end] = d.astype(np.float32)
        signs[start:end]     = np.where(mesh.contains(chunk), -1.0, 1.0)

    sdf = (distances * signs).reshape(resolution, resolution, resolution)
    return torch.tensor(sdf), bounds_min, bounds_max


def query_sdf(vertices: torch.Tensor, sdf_grid: torch.Tensor,
              bounds_min: np.ndarray, bounds_max: np.ndarray) -> torch.Tensor:
    bmin = torch.tensor(bounds_min, dtype=torch.float32, device=vertices.device)
    bmax = torch.tensor(bounds_max, dtype=torch.float32, device=vertices.device)

    coords = 2.0 * (vertices - bmin) / (bmax - bmin) - 1.0  # (V, 3)

    sdf_in = sdf_grid.permute(2, 1, 0).unsqueeze(0).unsqueeze(0).to(vertices.device)
    grid   = coords.view(1, 1, 1, -1, 3)   # (1, 1, 1, V, xyz)

    out = F.grid_sample(sdf_in, grid, mode="bilinear",
                        align_corners=True, padding_mode="border")
    return out.view(-1)  # (V,)

def build_laplacian(mesh: trimesh.Trimesh) -> torch.Tensor:
    n = len(mesh.vertices)
    edges = mesh.edges_unique       

    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(src, minlength=n).astype(np.float32)

    off_diag_vals = -1.0 / degree[src] 
    diag_vals     = np.ones(n, dtype=np.float32)

    rows = np.concatenate([src, np.arange(n)])
    cols = np.concatenate([dst, np.arange(n)])
    vals = np.concatenate([off_diag_vals, diag_vals])

    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

def mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return torch.abs((v0 * torch.cross(v1, v2, dim=1)).sum() / 6.0)


def optimize_mesh(
    original_mesh: trimesh.Trimesh, n_iters: int = 1000, lr: float = 1e-3, w_smooth: float = 1.0, w_inside: float = 100.0, w_volume: float = 0.1, sdf_resolution: int  = 64) -> trimesh.Trimesh:
    sdf_grid, bounds_min, bounds_max = build_sdf_grid(original_mesh, sdf_resolution)

    L = build_laplacian(original_mesh)

    faces = torch.tensor(original_mesh.faces, dtype=torch.long)
    verts = torch.tensor(original_mesh.vertices.copy(),
                         dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([verts], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iters)

    initial_vol = mesh_volume(verts.detach(), faces).item()

    for i in range(n_iters):
        optimizer.zero_grad()

        Lv = torch.sparse.mm(L, verts)     
        smooth_loss = (Lv ** 2).mean()

        sdf_vals = query_sdf(verts, sdf_grid, bounds_min, bounds_max)
        inside_loss = F.relu(sdf_vals).pow(2).mean()   

        vol = mesh_volume(verts, faces)
        vol_loss = -(vol / initial_vol)

        loss = w_smooth * smooth_loss + w_inside * inside_loss + w_volume * vol_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print(f"[{i:5d}/{n_iters}]  "
                  f"loss={loss.item():9.5f}  "
                  f"smooth={smooth_loss.item():.5f}  "
                  f"inside={inside_loss.item():.5f}  "
                  f"vol={vol.item():.5f}")

    final_verts = verts.detach().cpu().numpy()
    result = trimesh.Trimesh(vertices=final_verts,
                             faces=original_mesh.faces,
                             process=False)

    print(f"Final volume   : {mesh_volume(verts.detach(), faces).item():.6f}")
    return result


def save_single_view(mesh, rotation_matrix, filename):
    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy())
    mesh_copy.visual.vertex_colors = mesh.visual.vertex_colors
    s = trimesh.Scene(mesh)
    s.apply_transform(rotation_matrix)
    # png = s.save_image(resolution=[800,800], visible=True)
    # Image.open(io.BytesIO(png)).save(filename + ".png")

def save_views(mesh: trimesh.Trimesh):
    r_quarter = trimesh.transformations.rotation_matrix(np.pi/2.0, [0, 1, 0])
    r_half = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    r_three_quarter = trimesh.transformations.rotation_matrix(3.0*np.pi/2.0, [0, 1, 0])

    save_single_view(mesh, r_quarter, "quarter_view")
    save_single_view(mesh, r_half, "half_view")
    save_single_view(mesh, r_three_quarter, "three_quarter_view")

# ---- Helpers from (retired) main.py -----------------------------------------

from arlabelvis.colors import sRGBtoLAB
from arlabelvis.distances import euclidean_distance


def generate_LABs(stepSize: int = 16):
    """Generate the input RGB grid (at the given stepSize) and its CIELAB image.

    stepSize=1 -> dense 256^3 sampling (16.7M points); larger steps are used
    during exploration to keep MATLAB RGD tractable.
    """
    allRGBs = np.array([[r - 1, g - 1, b - 1] for r in range(0, 257, stepSize)
                                              for g in range(0, 257, stepSize)
                                              for b in range(0, 257, stepSize)])
    allRGBs = np.where(allRGBs < 0, 0, allRGBs)
    allRGBs = np.where(allRGBs > 255, 255, allRGBs)
    allLABs = sRGBtoLAB(allRGBs)
    return allRGBs, allLABs


def get_mesh_vertex_colors(mesh, allLABs, allRGBs):
    """Assign each mesh vertex the RGB of the nearest input LAB point."""
    colors = []
    for vert in mesh.vertices:
        best_idx = 0
        best_distance = euclidean_distance(vert, allLABs[0])
        for i, lab in enumerate(allLABs):
            distance = euclidean_distance(vert, lab)
            if distance < best_distance:
                best_idx = i
                best_distance = distance
        c = allRGBs[best_idx] / 255.0
        colors.append([c[0], c[1], c[2], 1.0])
    return np.array(colors)


def assign_vertex_colors(mesh: trimesh.Trimesh, allLABs, furthest):
    """Assign each mesh vertex the `furthest` RGB corresponding to the nearest LAB point."""
    colors = []
    for vertex in mesh.vertices:
        closest = 0
        closest_dist = 1e9
        for i, lab in enumerate(allLABs):
            d = euclidean_distance(lab, vertex)
            if d < closest_dist:
                closest = i
                closest_dist = d
        c = furthest[closest] / 255.0
        colors.append([c[0], c[1], c[2], 1.0])
    return np.array(colors)
