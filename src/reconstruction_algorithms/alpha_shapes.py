import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

DEBUG = True

def alpha_shapes_surface_reconstruction(point_cloud, alpha=1.0):
    """
    Perform surface reconstruction using the Alpha Shapes algorithm.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        alpha (float): The alpha parameter controlling the level of detail. Default is 1.0.

    Returns:
        o3d.geometry.TriangleMesh: The reconstructed surface mesh.
    """
    try:
        if DEBUG:
            print("Starting Alpha Shapes surface reconstruction...")

        # Validate the input point cloud
        if not isinstance(point_cloud, o3d.geometry.PointCloud):
            raise ValueError("Input must be an Open3D PointCloud object.")

        # Convert point cloud to numpy array
        points = np.asarray(point_cloud.points)
        if DEBUG:
            print(f"Point cloud contains {len(points)} points.")

        # Perform Delaunay triangulation
        from scipy.spatial import Delaunay
        delaunay = Delaunay(points)

        # Extract triangular faces from tetrahedra
        triangles_set = set()
        for simplex in delaunay.simplices:
            for i in range(4):
                face = tuple(sorted([simplex[j] for j in range(4) if j != i]))
                triangles_set.add(face)

        # Convert the set of triangles to a numpy array
        triangles = np.array(list(triangles_set), dtype=np.int32)
        if DEBUG:
            print(f"Extracted {len(triangles)} unique triangular faces from tetrahedra.")

        # Validate triangle indices
        triangles = np.array(triangles, dtype=np.int32)
        if DEBUG:
            print(f"Triangles array shape: {triangles.shape}, dtype: {triangles.dtype}")

        if triangles.size == 0:
            raise ValueError("No valid triangles generated. Check the alpha parameter or input point cloud.")

        max_index = triangles.max()
        if max_index >= len(points):
            raise ValueError(f"Invalid triangle indices detected. Max index {max_index} exceeds number of points {len(points)}.")

        # Filter triangles based on the alpha parameter
        filtered_triangles = []
        for triangle in triangles:
            # Calculate the circumradius of the triangle
            a, b, c = points[triangle]
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            s = (ab + bc + ca) / 2  # Semi-perimeter
            area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
            circumradius = (ab * bc * ca) / (4 * area) if area > 0 else float('inf')

            # Include the triangle if its circumradius is less than or equal to alpha
            if circumradius <= alpha:
                filtered_triangles.append(triangle)

        # Convert the filtered triangles to a numpy array
        filtered_triangles = np.array(filtered_triangles, dtype=np.int32)
        if DEBUG:
            print(f"Filtered {len(filtered_triangles)} triangles based on alpha={alpha}.")

        if filtered_triangles.size == 0:
            raise ValueError("No valid triangles remain after filtering. Check the alpha parameter or input point cloud.")

        # Create a mesh from the filtered triangles
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
        mesh.compute_vertex_normals()

        if DEBUG:
            print(f"Alpha Shapes generated {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces.")

        return mesh
    except Exception as e:
        if DEBUG:
            import traceback
            print("Error during Alpha Shapes surface reconstruction:")
            traceback.print_exc()
        raise