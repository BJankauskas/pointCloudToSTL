from src.utils import validate_point_cloud, filter_invalid_faces, visualize_point_cloud

from src.reconstruction_algorithms.delaunay import delaunay_surface_reconstruction
from src.reconstruction_algorithms.poisson import poisson_surface_reconstruction
from src.reconstruction_algorithms.marching_cubes import marching_cubes_surface_reconstruction
from src.reconstruction_algorithms.ball_pivoting import ball_pivoting_surface_reconstruction
from src.reconstruction_algorithms.alpha_shapes import alpha_shapes_surface_reconstruction
from src.reconstruction_algorithms.rbf import rbf_surface_reconstruction
from src.reconstruction_algorithms.voronoi import voronoi_surface_reconstruction
from src.reconstruction_algorithms.convex_hull import extract_outer_surface

DEBUG = True  # Set to True to enable debug output

# ...existing code for shared utilities or entry points...


