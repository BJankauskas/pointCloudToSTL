import numpy as np
import open3d as o3d

DEBUG = True

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

    return mesh