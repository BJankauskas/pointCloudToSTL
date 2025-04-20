import argparse
import open3d as o3d
import numpy as np
import logging
import signal
from point_cloud_processing import load_point_cloud, estimate_normals 
from surface_reconstruction import *
from stl_exporter import save_mesh_as_stl
from utils import check_and_correct_face_normals, check_normals, visualize_normals
from pre_postprocessing import (
    denoise_point_cloud,
    smooth_point_cloud,
    resample_point_cloud,
    smooth_mesh,
    fill_mesh_holes,
    simplify_mesh,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',  # Corrected format string
    datefmt='%H:%M:%S'
)

__version__ = "1.0.0"  # Project version

def get_parser():
    """
    Create and return the argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Surface Reconstruction")
    parser.add_argument("--cli-menu", type=bool, default="False",
                        help="Path to the input point cloud file.")
    parser.add_argument("--input_file_path", type=str, default="./data/example_point_cloud.xyz",
                        help="Path to the input point cloud file.")
    parser.add_argument("--algorithm", type=str, choices=["delaunay", "poisson", "convex_hull", "marching_cubes", "ball_pivoting", "alpha_shapes", "rbf", "voronoi", "mls"], default="delaunay",
                        help="Reconstruction algorithm to use: delaunay, poisson, convex_hull, marching_cubes, ball_pivoting, alpha_shapes, rbf, voronoi, or mls.")
    parser.add_argument("--visu_norms", type=str, default="False",
                        help="Visualize normals (True/False).")
    parser.add_argument("--poisson_depth", type=int, default=12,
                        help="Depth for Poisson surface reconstruction. " \
                        "Only applicable if algorithm is 'poisson'.")
    parser.add_argument("--density_percentile", type=int, default=30,
                        help="Density percentile for Poisson surface reconstruction. " \
                        "Only applicable if algorithm is 'poisson'.")
    parser.add_argument("--voxel_level", type=float, default=0.05,
                        help="Voxel resolution for Marching Cubes and Ball Pivoting surface reconstruction. " \
                        "Only applicable if algorithm is 'marching_cubes' and 'ball-pivoting'.")
    parser.add_argument("--output_stl_path", type=str, default="./data/outputs/mesh_output.stl",
                        help="Path to save the reconstructed mesh as STL.")
    parser.add_argument("--rbf_function", type=str, default="multiquadric",
                        help="RBF function to use for RBF surface reconstruction. " \
                        "Options: 'linear', 'cubic', 'quintic', 'thin_plate', 'multiquadric', 'inverse_multiquadric'. " \
                        "Only applicable if algorithm is 'rbf'.")
    parser.add_argument("--rbf_smooth", type=float, default=0.0,
                        help="Smoothing parameter for RBF surface reconstruction. " \
                        "Only applicable if algorithm is 'rbf'.")
    parser.add_argument(
        "--max-cores",
        type=int,
        default=1,
        help="Maximum number of CPU cores to use for parallel processing (default: use all available cores minus one). " \
             "Set to 1 to disable parallel processing."
    )
    parser.add_argument("--denoise_nb_neighbors", type=int, default=5,
                        help="Number of neighbors to analyze for point cloud denoising.")
    parser.add_argument("--denoise_std_ratio", type=float, default=0.5,
                        help="Standard deviation ratio for point cloud denoising.")
    parser.add_argument("--smooth_search_radius", type=float, default=0.05,
                        help="Search radius for point cloud smoothing.")
    parser.add_argument("--resample_voxel_size", type=float, default=0.025,
                        help="Voxel size for point cloud resampling.")
    parser.add_argument("--mesh_smooth_iterations", type=int, default=5,
                        help="Number of iterations for mesh smoothing.")
    parser.add_argument("--mesh_smooth_lambda", type=float, default=0.125,
                        help="Lambda value for mesh smoothing.")
    parser.add_argument("--max_hole_size", type=int, default=0.25,
                        help="Maximum hole size to fill in the mesh.")
    parser.add_argument("--simplify_target_reduction", type=float, default=0.75,
                        help="Target reduction fraction for mesh simplification.")
    parser.add_argument("--enable_denoising", type=str, default="True",
                        help="Enable or disable point cloud denoising (True/False).")
    parser.add_argument("--enable_simplification", type=str, default="True",
                        help="Enable or disable mesh simplification (True/False).")
    return parser

def validate_algorithm(algorithm):
    """
    Validate the provided algorithm name.
    """
    valid_algorithms = ["delaunay", "poisson", "convex_hull", "marching_cubes", "ball_pivoting", "alpha_shapes", "rbf", "voronoi", "mls"]
    if algorithm not in valid_algorithms:
        raise ValueError(f"Invalid algorithm '{algorithm}'. Valid options are: {', '.join(valid_algorithms)}")

def timeout_handler(signum, frame):
    raise TimeoutError("Surface reconstruction process took too long and was terminated.")

def cli_menu():
    """
    Display a multi-layered CLI menu to guide the user through algorithm selection and parameter configuration.
    """
    print("\n--- Surface Reconstruction CLI Menu ---\n")
    
    # Algorithm selection
    print("Select a reconstruction algorithm:")
    algorithms = ["delaunay", "poisson", "convex_hull", "marching_cubes", "ball_pivoting", "alpha_shapes", "rbf", "voronoi", "mls"]
    for i, algo in enumerate(algorithms, 1):
        print(f"{i}. {algo}")
    algo_choice = int(input("Enter the number corresponding to your choice: "))
    algorithm = algorithms[algo_choice - 1]

    # Preprocessing menu
    enable_preprocessing = input("Enable preprocessing? (True/False): ").strip().lower() in ["true", "1", "yes"]
    if enable_preprocessing:
        print("\n--- Preprocessing Configuration ---")
        enable_denoising = input("Enable point cloud denoising? (True/False): ").strip().lower() in ["true", "1", "yes"]
        enable_smoothing = input("Enable point cloud smoothing? (True/False): ").strip().lower() in ["true", "1", "yes"]
        enable_resampling = input("Enable point cloud resampling? (True/False): ").strip().lower() in ["true", "1", "yes"]

        # Preprocessing parameters
        denoise_nb_neighbors = int(input("Enter number of neighbors for denoising (default 20): ") or 20)
        denoise_std_ratio = float(input("Enter standard deviation ratio for denoising (default 2.0): ") or 2.0)
        smooth_search_radius = float(input("Enter search radius for smoothing (default 0.1): ") or 0.1)
        resample_voxel_size = float(input("Enter voxel size for resampling (default 0.05): ") or 0.05)
    else:
        enable_denoising = enable_smoothing = enable_resampling = False
        denoise_nb_neighbors = denoise_std_ratio = smooth_search_radius = resample_voxel_size = None

    # Postprocessing menu
    enable_postprocessing = input("Enable postprocessing? (True/False): ").strip().lower() in ["true", "1", "yes"]
    if enable_postprocessing:
        print("\n--- Postprocessing Configuration ---")
        enable_simplification = input("Enable mesh simplification? (True/False): ").strip().lower() in ["true", "1", "yes"]
        enable_hole_filling = input("Enable mesh hole filling? (True/False): ").strip().lower() in ["true", "1", "yes"]

        # Postprocessing parameters
        simplify_target_reduction = float(input("Enter target reduction for mesh simplification (default 0.5): ") or 0.5)
        max_hole_size = int(input("Enter maximum hole size for filling (default 10): ") or 10)
        mesh_smooth_iterations = int(input("Enter number of iterations for mesh smoothing (default 5): ") or 5)
        mesh_smooth_lambda = float(input("Enter lambda value for mesh smoothing (default 0.125): ") or 0.125)
    else:
        enable_simplification = enable_hole_filling = False
        simplify_target_reduction = max_hole_size = mesh_smooth_iterations = mesh_smooth_lambda = None

    # Reconstruction parameters
    print("\n--- Reconstruction Configuration ---")
    voxel_level = float(input("Enter voxel resolution (default 0.05): ") or 0.05)
    poisson_depth = int(input("Enter Poisson reconstruction depth (default 12): ") or 12)
    density_percentile = int(input("Enter density percentile for Poisson reconstruction (default 30): ") or 30)

    return {
        "algorithm": algorithm,
        "enable_preprocessing": enable_preprocessing,
        "enable_denoising": enable_denoising,
        "enable_smoothing": enable_smoothing,
        "enable_resampling": enable_resampling,
        "denoise_nb_neighbors": denoise_nb_neighbors,
        "denoise_std_ratio": denoise_std_ratio,
        "smooth_search_radius": smooth_search_radius,
        "resample_voxel_size": resample_voxel_size,
        "enable_postprocessing": enable_postprocessing,
        "enable_simplification": enable_simplification,
        "enable_hole_filling": enable_hole_filling,
        "simplify_target_reduction": simplify_target_reduction,
        "max_hole_size": max_hole_size,
        "mesh_smooth_iterations": mesh_smooth_iterations,
        "mesh_smooth_lambda": mesh_smooth_lambda,
        "voxel_level": voxel_level,
        "poisson_depth": poisson_depth,
        "density_percentile": density_percentile,
    }

def main():
    # Add a flag to choose between CLI menu or command-line arguments
    use_cli_menu = input("Use CLI menu for configuration? (True/False): ").strip().lower() in ["true", "1", "yes"]

    if use_cli_menu:
        # Display CLI menu
        user_config = cli_menu()

        # Load input file
        input_file = input("Enter the path to the input point cloud file: ").strip()
        output_file = input("Enter the path to save the output STL file: ").strip()

        # Validate the algorithm
        try:
            validate_algorithm(user_config["algorithm"])
        except ValueError as e:
            logging.error(e)
            return
    else:
        # Use command-line arguments
        parser = get_parser()
        args = parser.parse_args()

        # Validate the algorithm
        try:
            validate_algorithm(args.algorithm)
        except ValueError as e:
            logging.error(e)
            return

        user_config = {
            "algorithm": args.algorithm,
            "enable_preprocessing": True,
            "enable_denoising": args.enable_denoising.lower() in ["true", "1", "yes"],
            "enable_smoothing": True,
            "enable_resampling": True,
            "denoise_nb_neighbors": args.denoise_nb_neighbors,
            "denoise_std_ratio": args.denoise_std_ratio,
            "smooth_search_radius": args.smooth_search_radius,
            "resample_voxel_size": args.resample_voxel_size,
            "enable_postprocessing": True,
            "enable_simplification": args.enable_simplification.lower() in ["true", "1", "yes"],
            "enable_hole_filling": True,
            "simplify_target_reduction": args.simplify_target_reduction,
            "max_hole_size": args.max_hole_size,
            "mesh_smooth_iterations": args.mesh_smooth_iterations,
            "mesh_smooth_lambda": args.mesh_smooth_lambda,
            "voxel_level": args.voxel_level,
            "poisson_depth": args.poisson_depth,
            "density_percentile": args.density_percentile,
        }

        input_file = args.input_file_path
        output_file = args.output_stl_path

    try:
        logging.info("Loading point cloud...")
        point_cloud = load_point_cloud(input_file)
    except Exception as e:
        logging.error(f"Error loading point cloud: {e}")
        return

    try:
        logging.info("Estimating normals...")
        point_cloud = estimate_normals(point_cloud)

        logging.info("Verifying normals after estimation...")
        if point_cloud.has_normals():
            logging.info("Normals are present after estimation.")
        else:
            logging.info("Normals are missing after estimation.")

        logging.info("Checking normals...")
        check_normals(point_cloud)

        # Preprocessing
        if user_config["enable_preprocessing"]:
            if user_config["enable_denoising"]:
                logging.info("Denoising point cloud...")
                point_cloud = denoise_point_cloud(point_cloud, nb_neighbors=user_config["denoise_nb_neighbors"], std_ratio=user_config["denoise_std_ratio"])

            if user_config["enable_smoothing"]:
                logging.info("Smoothing point cloud...")
                point_cloud = smooth_point_cloud(point_cloud, search_radius=user_config["smooth_search_radius"])

            if user_config["enable_resampling"]:
                logging.info("Resampling point cloud...")
                point_cloud = resample_point_cloud(point_cloud, voxel_size=user_config["resample_voxel_size"])

        # Recompute normals after preprocessing
        logging.info("Recomputing normals after preprocessing...")
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=user_config["voxel_level"], max_nn=30))
        point_cloud.orient_normals_consistent_tangent_plane(k=30)

        # Reconstruction
        if user_config["algorithm"] == "poisson":
            logging.info("Performing Poisson surface reconstruction...")
            mesh = poisson_surface_reconstruction(point_cloud, user_config["poisson_depth"], user_config["density_percentile"])
        # Add other algorithms as needed...

        # Postprocessing
        if user_config["enable_postprocessing"]:
            if user_config["enable_hole_filling"]:
                logging.info("Filling holes in the mesh...")
                mesh = fill_mesh_holes(mesh, max_hole_size=user_config["max_hole_size"])

            if user_config["enable_simplification"]:
                logging.info("Simplifying mesh...")
                mesh = simplify_mesh(mesh, target_reduction=user_config["simplify_target_reduction"])

            logging.info("Smoothing mesh...")
            mesh = smooth_mesh(mesh, iterations=user_config["mesh_smooth_iterations"], lambda_filter=user_config["mesh_smooth_lambda"])

        # Save the mesh
        logging.info("Saving mesh as STL...")
        save_mesh_as_stl(mesh, output_file)

        logging.info("Surface reconstruction completed.")
    except Exception as e:
        logging.error(f"Error during surface reconstruction: {e}")

if __name__ == "__main__":
    main()
