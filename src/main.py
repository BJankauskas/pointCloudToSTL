import argparse
import open3d as o3d
import numpy as np
import logging
from point_cloud_processing import load_point_cloud, estimate_normals 
from surface_reconstruction import delaunay_surface_reconstruction, poisson_surface_reconstruction, extract_outer_surface
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
    parser.add_argument("--algorithm", type=str, choices=["delaunay", "poisson", "convex_hull"], default="delaunay",
                        help="Reconstruction algorithm to use: delaunay, poisson, or convex_hull.")
    parser.add_argument("--visu_norms", type=str, default="False",
                        help="Visualize normals (True/False).")
    parser.add_argument("--poisson_depth", type=int, default=12,
                        help="Depth for Poisson surface reconstruction.")
    parser.add_argument("--density_percentile", type=int, default=30,
                        help="Density percentile for Poisson surface reconstruction.")
    parser.add_argument("--output_stl_path", type=str, default="./data/outputs/mesh_output.stl",
                        help="Path to save the reconstructed mesh as STL.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input_file_path  # Corrected argument name
    output_file = args.output_stl_path  # Use dynamic output path

    # Convert visu_norms to boolean
    visu_norms = args.visu_norms.lower() in ["true", "1", "yes"]

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
            mesh = poisson_surface_reconstruction(point_cloud)
        elif args.algorithm == "convex_hull":
            logging.info("Extracting outer surface using convex hull...")
            points = np.asarray(point_cloud.points)
            faces = extract_outer_surface(points)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()

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
