import open3d as o3d

DEBUG = False  # Set to True to enable debug output

def load_point_cloud(file_path):
    """
    Load a point cloud from a file.
    Supported formats depend on Open3D (e.g., .ply, .xyz, .pcd).
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    if point_cloud.is_empty():
        raise ValueError("Loaded point cloud is empty. Check the file path or format.")
    if DEBUG:
        print(f"Loaded point cloud with {len(point_cloud.points)} points.")
    return point_cloud

def estimate_normals(point_cloud, radius=0.1, max_nn=30):
    """
    Estimate normals for the point cloud.
    
    Parameters:
        point_cloud (open3d.geometry.PointCloud): Point cloud object.
        radius (float): Radius of the sphere used for normal estimation.
        max_nn (int): Maximum number of nearest neighbors used for normal estimation.
    """
    try:
        if point_cloud is None or point_cloud.is_empty():
            raise ValueError("Invalid or empty point cloud. Cannot estimate normals.")
        if DEBUG:
            print("Estimating normals for the point cloud...")
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        if DEBUG:
            print("Normals estimated successfully.")
        # Verify that normals are set
        if not point_cloud.has_normals():
            raise ValueError("Normals could not be estimated. Check the input point cloud or parameters.")
    except Exception as e:
        print(f"Error estimating normals: {e}")
    return point_cloud

# Example usage (for testing):
if __name__ == "__main__":
    # Replace 'example_point_cloud.ply' with your actual file path
    file_path = "../data/inputData/example_point_cloud.xyz"
    point_cloud = load_point_cloud(file_path)
    if point_cloud:
        estimate_normals(point_cloud)
