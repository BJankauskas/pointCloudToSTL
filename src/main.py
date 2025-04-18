import argparse
import open3d as o3d
import numpy as np
import logging
import signal
from point_cloud_processing import load_point_cloud, estimate_normals 
from surface_reconstruction import delaunay_surface_reconstruction, poisson_surface_reconstruction, extract_outer_surface, marching_cubes_surface_reconstruction
from stl_exporter import save_mesh_as_stl
from utils import check_and_correct_face_normals, check_normals, visualize_normals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

__version__ = "1.0.0"  # Project version

def get_parser():
    """
    Create and return the argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Surface Reconstruction")
    parser.add_argument("--input_file_path", type=str, default="./data/example_point_cloud.xyz",
                        help="Path to the input point cloud file.")
    parser.add_argument("--algorithm", type=str, choices=["delaunay", "poisson", "convex_hull", "marching_cubes"], default="delaunay",
                        help="Reconstruction algorithm to use: delaunay, poisson, convex_hull, or marching_cubes.")
    parser.add_argument("--visu_norms", type=str, default="False",
                        help="Visualize normals (True/False).")
    parser.add_argument("--poisson_depth", type=int, default=12,
                        help="Depth for Poisson surface reconstruction. Only applicable if algorithm is 'poisson'.")
    parser.add_argument("--density_percentile", type=int, default=30,
                        help="Density percentile for Poisson surface reconstruction. Only applicable if algorithm is 'poisson'.")
    parser.add_argument("--voxel_level", type=float, default=0.05,
                        help="Voxel resolution for Marching Cubes surface reconstruction. Only applicable if algorithm is 'marching_cubes'.")
    parser.add_argument("--output_stl_path", type=str, default="./data/outputs/mesh_output.stl",
                        help="Path to save the reconstructed mesh as STL.")
    return parser

def timeout_handler(signum, frame):
    raise TimeoutError("Surface reconstruction process took too long and was terminated.")

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input_file_path  # Corrected argument name
    output_file = args.output_stl_path  # Use dynamic output path

    # Convert visu_norms to boolean
    visu_norms = args.visu_norms.lower() in ["true", "1", "yes"]

    # Set a timeout for the surface reconstruction process
    signal.signal(signal.SIGALRM, timeout_handler)
    reconstruction_timeout = 300  # Timeout in seconds (e.g., 5 minutes)
    signal.alarm(reconstruction_timeout)

    try:
        logging.info("Loading point cloud...")
        point_cloud = load_point_cloud(input_file)
    except Exception as e:
        logging.error(f"Error loading point cloud: {e}")
        return

    try:
        logging.info("Estimating normals...")
        point_cloud = estimate_normals(point_cloud)

        #logging.info(f"Type of point_cloud after estimation: {type(point_cloud)}")
        #logging.info(f"Point cloud has {len(point_cloud.points)} points after estimation.")

        logging.info("Verifying normals after estimation...")
        if point_cloud.has_normals():
            logging.info("Normals are present after estimation.")
        else:
            logging.info("Normals are missing after estimation.")

        logging.info("Checking normals...")
        check_normals(point_cloud)

        if visu_norms:
            logging.info("Visualizing normals...")
            visualize_normals(point_cloud)

        if args.algorithm == "delaunay":
            logging.info("Performing 3D Delaunay surface reconstruction...")
            mesh = delaunay_surface_reconstruction(point_cloud)
        elif args.algorithm == "poisson":
            logging.info("Performing Poisson surface reconstruction...")
            mesh = poisson_surface_reconstruction(point_cloud, args.depth, args.density_percentile)
        elif args.algorithm == "convex_hull":
            logging.info("Extracting outer surface using convex hull...")
            points = np.asarray(point_cloud.points)
            faces = extract_outer_surface(points)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
        elif args.algorithm == "marching_cubes":
            logging.info("Performing Marching Cubes surface reconstruction...")
            # Increase voxel grid resolution by reducing voxel size
            voxel_size = args.voxel_level  # Reduced voxel size for higher resolution
            bounding_box = point_cloud.get_axis_aligned_bounding_box()
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
                point_cloud, voxel_size=voxel_size, min_bound=bounding_box.min_bound, max_bound=bounding_box.max_bound
            )

            # Calculate the extent of the voxel grid manually
            min_bound = voxel_grid.get_min_bound()
            max_bound = voxel_grid.get_max_bound()
            extent = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

            # Convert voxel grid to volumetric data with interpolation
            volume_data = np.zeros(extent, dtype=np.float32)
            for voxel in voxel_grid.get_voxels():
                x, y, z = voxel.grid_index
                volume_data[x, y, z] = 1  # Mark occupied voxels

            # Apply Gaussian smoothing to interpolate and fill gaps
            from scipy.ndimage import gaussian_filter
            volume_data = gaussian_filter(volume_data, sigma=1)

            mesh = marching_cubes_surface_reconstruction(volume_data, voxel_size)

        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            logging.warning("Warning: Reconstructed mesh is empty. Skipping STL export.")
        else:
            check_and_correct_face_normals(mesh)
            logging.info("Saving mesh as STL...")
            save_mesh_as_stl(mesh, output_file)

        logging.info("Surface reconstruction completed.")
    except Exception as e:
        logging.error(f"Error during surface reconstruction: {e}")

if __name__ == "__main__":
    main()
