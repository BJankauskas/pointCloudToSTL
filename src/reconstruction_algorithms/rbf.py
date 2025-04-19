import numpy as np
import open3d as o3d
import scipy.interpolate as si

DEBUG = True

def rbf_surface_reconstruction(point_cloud, function='multiquadric', smooth=0, max_cores=1):
    """
    Perform surface reconstruction using Radial Basis Function (RBF) interpolation.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        function (str): The RBF function to use ('multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate').
        smooth (float): Smoothing parameter for RBF interpolation. Default is 0 (no smoothing).
        max_cores (int): This parameter is now ignored as parallel processing is removed.

    Returns:
        o3d.geometry.TriangleMesh: The reconstructed surface mesh.
    """
    if DEBUG:
        print("Starting RBF surface reconstruction with sequential processing...")

    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    if DEBUG:
        print(f"Point cloud contains {len(points)} points.")

    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Create a grid for interpolation
    grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]

    # Perform RBF interpolation sequentially
    rbf = si.Rbf(x, y, z, function=function, smooth=smooth)
    grid_z = rbf(grid_x, grid_y)

    # Reshape the result to match the grid
    grid_z = np.array(grid_z).reshape(grid_x.shape)

    # Create a mesh from the interpolated grid
    vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    faces = []
    for i in range(grid_x.shape[0] - 1):
        for j in range(grid_x.shape[1] - 1):
            v1 = i * grid_x.shape[1] + j
            v2 = v1 + 1
            v3 = v1 + grid_x.shape[1]
            v4 = v3 + 1
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    # Convert to Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
    mesh.compute_vertex_normals()

    if DEBUG:
        print(f"RBF surface reconstruction completed with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces.")

    return mesh