import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

DEBUG = True

def marching_cubes_surface_reconstruction(volume_data, level=0.0):
    """
    Perform surface reconstruction using the Marching Cubes algorithm.

    Parameters:
        volume_data (numpy.ndarray): 3D volumetric data array.
        level (float): The level value to extract the surface. Default is 0.0.

    Returns:
        o3d.geometry.TriangleMesh: The reconstructed surface mesh.
    """
    if DEBUG:
        print("Starting Marching Cubes surface reconstruction...")

    # Validate the input volume data
    if not isinstance(volume_data, np.ndarray) or volume_data.ndim != 3:
        raise ValueError("Input volume_data must be a 3D numpy array.")

    # Determine a valid level dynamically based on the range of the volumetric data
    min_value, max_value = np.min(volume_data), np.max(volume_data)
    level = (min_value + max_value) / 2  # Set level to the midpoint of the range

    if DEBUG:
        print(f"Volume data range: min={min_value}, max={max_value}, selected level={level}")

    # Perform Marching Cubes algorithm
    vertices, faces, normals, _ = marching_cubes(volume_data, level=level)

    if DEBUG:
        print(f"Marching Cubes generated {len(vertices)} vertices and {len(faces)} faces.")

    # Create a mesh from the vertices and faces
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # Ensure consistent face orientation
    mesh.orient_triangles()
    mesh.compute_triangle_normals()

    if DEBUG:
        print("Marching Cubes surface reconstruction completed.")

    return mesh