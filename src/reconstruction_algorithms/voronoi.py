import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Voronoi

DEBUG = True

def voronoi_surface_reconstruction(point_cloud):
    """
    Perform surface reconstruction using Voronoi diagrams.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.

    Returns:
        o3d.geometry.TriangleMesh: The reconstructed surface mesh.
    """
    if DEBUG:
        print("Starting Voronoi-based surface reconstruction...")

    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)
    if DEBUG:
        print(f"Point cloud contains {len(points)} points.")

    # Check if the point cloud has enough points for Voronoi reconstruction
    if len(points) < 4:
        raise ValueError("Voronoi-based reconstruction requires at least 4 points.")

    # Compute the Voronoi diagram
    vor = Voronoi(points)
    if DEBUG:
        print(f"Voronoi diagram computed with {len(vor.vertices)} vertices.")
        print(f"Voronoi ridge vertices: {vor.ridge_vertices}")

    # Compute the convex hull of the input points
    hull = ConvexHull(points)
    hull_min = points[hull.vertices].min(axis=0)
    hull_max = points[hull.vertices].max(axis=0)

    # Extract vertices and ridge vertices
    vertices = vor.vertices
    ridge_vertices = vor.ridge_vertices

    # Filter valid ridges (include ridges that partially intersect the convex hull bounds)
    valid_faces = []
    for ridge in ridge_vertices:
        if -1 in ridge:
            # Handle unbounded ridges by approximating them
            bounded_ridge = [v for v in ridge if v != -1]
            if len(bounded_ridge) == 2:  # Only consider ridges with two bounded vertices
                midpoint = (vertices[bounded_ridge[0]] + vertices[bounded_ridge[1]]) / 2
                if np.all((midpoint >= hull_min) & (midpoint <= hull_max)):
                    valid_faces.append(bounded_ridge)
        else:
            # Check if all vertices of the ridge are within the convex hull bounds
            triangle = np.array(ridge)
            if np.all((vertices[triangle] >= hull_min) & (vertices[triangle] <= hull_max)):
                if len(triangle) == 3:  # Ensure the ridge forms a triangle
                    valid_faces.append(triangle)

    if DEBUG:
        print(f"Extracted {len(valid_faces)} valid triangular faces.")
        if len(valid_faces) == 0:
            print("No valid faces found. Debugging Voronoi vertices and ridges:")
            print(f"Voronoi vertices:\n{vertices}")
            print(f"Ridge vertices:\n{ridge_vertices}")

    # Handle cases where no valid faces are found
    if len(valid_faces) == 0:
        raise ValueError("No valid triangular faces were found during Voronoi-based reconstruction. Ensure the input point cloud is sufficiently dense.")

    # Filter vertices to include only those used in valid faces
    used_vertex_indices = set(idx for face in valid_faces for idx in face)
    filtered_vertices = vertices[list(used_vertex_indices)]
    vertex_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices)}

    # Remap faces to the filtered vertex indices
    remapped_faces = [[vertex_index_map[idx] for idx in face] for face in valid_faces]

    # Create a mesh from the filtered vertices and remapped faces
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(remapped_faces, dtype=np.int32))
    mesh.compute_vertex_normals()

    if DEBUG:
        print(f"Voronoi-based surface reconstruction completed with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces.")

    return mesh