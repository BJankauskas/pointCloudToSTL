import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, ConvexHull
from src.utils import validate_point_cloud, filter_invalid_faces, visualize_point_cloud

DEBUG = False  # Set to True to enable debug output
 

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

def poisson_surface_reconstruction(point_cloud, depth=8, density_percentile=10):
    """
    Perform Poisson surface reconstruction on the point cloud and trim the mesh using the density field.
    Ensure the trimmed surface matches the point cloud extents.
    """
    if DEBUG:
        print("Starting Poisson surface reconstruction...")

    # Debugging: Check if normals are present before reconstruction
    if not point_cloud.has_normals():
        raise ValueError("Point cloud does not have normals before Poisson reconstruction.")
    if DEBUG:
        print("Normals are present before Poisson reconstruction.")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth
    )
    if DEBUG:
        print(f"Reconstructed mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces before trimming.")

    # Convert densities to a numpy array
    densities = np.asarray(densities)

    # Compute a dynamic density threshold based on the specified percentile
    density_threshold = np.percentile(densities, density_percentile)
    if DEBUG:
        print(f"Using density threshold: {density_threshold}")

    # Trim the mesh based on the density threshold
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Apply bounding box filtering to ensure the surface matches the point cloud extents
    bounding_box = point_cloud.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bounding_box)

    if DEBUG:
        print(f"Reconstructed mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces after trimming and bounding box filtering.")

    # Save a visualization of the trimmed mesh
    #trimmed_mesh_path = "./data/trimmed_mesh_visualization.ply"
    #o3d.io.write_triangle_mesh(trimmed_mesh_path, mesh)
    #if DEBUG:
    #    print(f"Trimmed mesh visualization saved to {trimmed_mesh_path}")

    return mesh

def extract_outer_surface(points):
    """
    Extract the outer surface of the point cloud using the convex hull algorithm.
    """
    if DEBUG:
        print("Extracting outer surface using convex hull...")
    hull = ConvexHull(points)
    faces = hull.simplices
    if DEBUG:
        print(f"Extracted {len(faces)} outer faces.")
    return faces

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
    from skimage.measure import marching_cubes
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


