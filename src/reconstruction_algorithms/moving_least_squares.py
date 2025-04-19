import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

DEBUG = True

def moving_least_squares_surface_reconstruction(point_cloud, search_radius=0.1):
    """
    Perform surface reconstruction using the Moving Least Squares (MLS) algorithm.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        search_radius (float): The radius used to search for neighboring points. Default is 0.1.

    Returns:
        o3d.geometry.PointCloud: The smoothed point cloud after applying MLS.
    """
    if DEBUG:
        print("Starting Moving Least Squares (MLS) surface reconstruction...")

    # Validate the input point cloud
    if not isinstance(point_cloud, o3d.geometry.PointCloud):
        raise ValueError("Input must be an Open3D PointCloud object.")

    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    if DEBUG:
        print(f"Point cloud contains {len(points)} points.")

    # Build a KDTree for efficient neighbor search
    kdtree = KDTree(points)

    # Initialize an array to store smoothed points
    smoothed_points = []

    # Perform MLS for each point
    for i, point in enumerate(points):
        # Find neighbors within the search radius
        indices = kdtree.query_ball_point(point, r=search_radius)
        if len(indices) < 3:  # Skip points with insufficient neighbors
            smoothed_points.append(point)
            continue

        neighbors = points[indices]

        # Compute the centroid of the neighbors
        centroid = np.mean(neighbors, axis=0)

        # Perform weighted least squares fitting (plane fitting)
        weights = np.exp(-np.linalg.norm(neighbors - point, axis=1) ** 2 / (2 * search_radius ** 2))
        A = neighbors - centroid
        W = np.diag(weights)
        covariance_matrix = A.T @ W @ A
        _, _, vh = np.linalg.svd(covariance_matrix)
        normal = vh[-1]

        # Project the point onto the fitted plane
        projection = point - np.dot(point - centroid, normal) * normal
        smoothed_points.append(projection)

        if DEBUG and i % 100 == 0:
            print(f"Processed {i}/{len(points)} points...")

    # Create a new point cloud with smoothed points
    smoothed_point_cloud = o3d.geometry.PointCloud()
    smoothed_point_cloud.points = o3d.utility.Vector3dVector(np.array(smoothed_points))

    if DEBUG:
        print("MLS surface reconstruction completed.")

    return smoothed_point_cloud
