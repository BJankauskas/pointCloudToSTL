import open3d as o3d
import numpy as np

DEBUG = True

def denoise_point_cloud(point_cloud, nb_neighbors=20, std_ratio=2.0, enable_denoising=True):
    """
    Remove noise and outliers from the point cloud using statistical outlier removal.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        nb_neighbors (int): Number of neighbors to analyze for each point.
        std_ratio (float): Standard deviation ratio for filtering.
        enable_denoising (bool): Whether to enable denoising.

    Returns:
        o3d.geometry.PointCloud: The denoised point cloud.
    """
    if not enable_denoising:
        if DEBUG:
            print("Denoising is disabled. Returning the original point cloud.")
        return point_cloud

    if DEBUG:
        print("Starting point cloud denoising...")
    denoised_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if DEBUG:
        print(f"Point cloud denoising completed. Remaining points: {len(denoised_cloud.points)}")
    return denoised_cloud

def smooth_point_cloud(point_cloud, search_radius=0.1):
    """
    Smooth the point cloud using a manual Moving Least Squares (MLS) implementation.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        search_radius (float): The radius used to search for neighboring points.

    Returns:
        o3d.geometry.PointCloud: The smoothed point cloud.
    """
    if DEBUG:
        print("Starting point cloud smoothing...")

    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    if DEBUG:
        print(f"Point cloud contains {len(points)} points.")

    # Build a KDTree for efficient neighbor search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    # Initialize an array to store smoothed points
    smoothed_points = []

    # Perform MLS for each point
    for i, point in enumerate(points):
        # Find neighbors within the search radius
        [_, idx, _] = kdtree.search_radius_vector_3d(point, search_radius)
        neighbors = points[idx]

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
        print("Point cloud smoothing completed.")

    return smoothed_point_cloud

def resample_point_cloud(point_cloud, voxel_size=0.05):
    """
    Resample the point cloud to ensure uniform density using voxel downsampling.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The size of the voxel grid.

    Returns:
        o3d.geometry.PointCloud: The resampled point cloud.
    """
    if DEBUG:
        print("Starting point cloud resampling...")
    resampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    if DEBUG:
        print(f"Point cloud resampling completed. Remaining points: {len(resampled_cloud.points)}")
    return resampled_cloud

def smooth_mesh(mesh, iterations=10, lambda_filter=0.5):
    """
    Smooth the mesh using Laplacian smoothing.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        iterations (int): Number of smoothing iterations.
        lambda_filter (float): Smoothing factor.

    Returns:
        o3d.geometry.TriangleMesh: The smoothed mesh.
    """
    if DEBUG:
        print("Starting mesh smoothing...")
    smoothed_mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=iterations,
        lambda_filter=lambda_filter,
        filter_scope=o3d.geometry.FilterScope.All
    )
    if DEBUG:
        print("Mesh smoothing completed.")
    return smoothed_mesh

def fill_mesh_holes(mesh, max_hole_size=10):
    """
    Fill small holes in the mesh by identifying boundary edges and closing them.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        max_hole_size (int): Maximum number of boundary edges to consider as a hole.

    Returns:
        o3d.geometry.TriangleMesh: The mesh with small holes filled.
    """
    if DEBUG:
        print("Starting hole filling in the mesh...")

    # Identify boundary edges
    boundary_edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    if DEBUG:
        print(f"Found {len(boundary_edges)} boundary edges.")

    # Attempt to close small holes
    for boundary in boundary_edges:
        if len(boundary) <= max_hole_size:
            # Close the hole by adding a new face
            vertices = [mesh.vertices[i] for i in boundary]
            centroid = np.mean(vertices, axis=0)
            centroid_idx = len(mesh.vertices)
            mesh.vertices.append(centroid)
            for i in range(len(boundary)):
                mesh.triangles.append([boundary[i], boundary[(i + 1) % len(boundary)], centroid_idx])

    if DEBUG:
        print("Hole filling completed.")
    return mesh

def simplify_mesh(mesh, target_reduction=0.5, enable_simplification=True):
    """
    Simplify the mesh while preserving its overall shape.

    Parameters:
        mesh (o3d.geometry.TriangleMesh): The input mesh.
        target_reduction (float): Fraction of triangles to retain (0.0 to 1.0).
        enable_simplification (bool): Whether to enable mesh simplification.

    Returns:
        o3d.geometry.TriangleMesh: The simplified mesh.
    """
    if not enable_simplification:
        if DEBUG:
            print("Mesh simplification is disabled. Returning the original mesh.")
        return mesh

    if DEBUG:
        print("Starting mesh simplification...")
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(len(mesh.triangles) * target_reduction))
    if DEBUG:
        print(f"Mesh simplification completed. Remaining triangles: {len(simplified_mesh.triangles)}")
    return simplified_mesh
