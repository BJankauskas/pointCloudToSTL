import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from src.utils import validate_point_cloud, filter_invalid_faces

DEBUG = True

def delaunay_surface_reconstruction(point_cloud):
    """
    Perform 3D Delaunay triangulation on the point cloud and create a mesh.
    """
    if DEBUG:
        print("Starting 3D Delaunay surface reconstruction...")
    points = np.asarray(point_cloud.points)

    # Validate the point cloud
    points = validate_point_cloud(points)

    # Debugging: Check if normals are present before reconstruction
    if not point_cloud.has_normals():
        raise ValueError("Point cloud does not have normals before Delaunay reconstruction.")
    if DEBUG:
        print("Normals are present before Delaunay reconstruction.")

    try:
        # Perform Delaunay triangulation
        delaunay = Delaunay(points)
        faces = delaunay.simplices

        # Debugging: Check the number of faces generated
        if DEBUG:
            print(f"Delaunay triangulation generated {len(faces)} faces.")

        if len(faces) == 0:
            raise ValueError("Delaunay triangulation did not produce any faces. Check the input point cloud.")

        # Validate faces before creating the mesh
        faces = filter_invalid_faces(points, faces)
        if DEBUG:
            print(f"Number of valid faces after filtering: {len(faces)}")

        if len(faces) == 0:
            raise ValueError("No valid faces remain after filtering. Cannot create a mesh.")

        if DEBUG:
            print(f"Reconstructed mesh has {len(points)} vertices and {len(faces)} faces.")

        # Create a mesh from the triangulation
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()

        # Ensure consistent face orientation
        if DEBUG:
            print("Ensuring consistent face orientation...")
        mesh.orient_triangles()
        mesh.compute_triangle_normals()
        if DEBUG:
            print("Face orientation has been corrected.")

        # Recompute vertex and triangle normals to ensure consistency
        if DEBUG:
            print("Recomputing vertex and triangle normals...")
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        if DEBUG:
            print("Normals recomputed successfully.")

        # Debugging: Check mesh properties after orientation
        if DEBUG:
            print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles after orientation.")
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            raise ValueError("Mesh is empty after orientation. Check the input data and processing steps.")

        return mesh
    except Exception as e:
        if DEBUG:
            print(f"Error during Delaunay triangulation: {e}")
        return o3d.geometry.TriangleMesh()