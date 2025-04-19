import numpy as np
import open3d as o3d

DEBUG = True

def ball_pivoting_surface_reconstruction(point_cloud, radius=0.05):
    """
    Perform surface reconstruction using the Ball Pivoting algorithm.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        radius (float): The radius of the ball used for pivoting. Default is 0.05.

    Returns:
        o3d.geometry.TriangleMesh: The reconstructed surface mesh.
    """
    if DEBUG:
        print("Starting Ball Pivoting surface reconstruction...")

    # Validate the input point cloud
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("Input must be an Open3D PointCloud object.")

    # Estimate normals if not already present
    if not point_cloud.has_normals():
        if DEBUG:
            print("Estimating normals for the point cloud...")
        point_cloud.estimate_normals()

    # Perform Ball Pivoting algorithm
    radii = [radius, radius * 1.5, radius * 2.0]  # Use multiple radii for better results
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud, o3d.utility.DoubleVector(radii)
    )

    if DEBUG:
        print(f"Ball Pivoting generated {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces.")

    # Ensure consistent face orientation
    mesh.orient_triangles()
    mesh.compute_triangle_normals()

    if DEBUG:
        print("Ball Pivoting surface reconstruction completed.")

    return mesh