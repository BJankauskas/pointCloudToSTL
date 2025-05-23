import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DEBUG = False  # Set to True to enable debug output

# Utility functions
def validate_file(file_path):
    pass

def validate_point_cloud(points):
    """
    Remove duplicate points from the point cloud.
    """
    unique_points = np.unique(points, axis=-1)
    if len(unique_points) < len(points):
        if DEBUG:
            print(f"Removed {len(points) - len(unique_points)} duplicate points.")
    return unique_points

def filter_invalid_faces(points, faces):
    """
    Extract triangular faces from tetrahedra generated by Delaunay triangulation.
    """
    if DEBUG:
        print(f"Total tetrahedra before filtering: {len(faces)}")
    triangular_faces = set()
    for tetrahedron in faces:
        # Extract all four triangular faces from the tetrahedron
        triangular_faces.add(tuple(sorted([tetrahedron[0], tetrahedron[1], tetrahedron[2]])))
        triangular_faces.add(tuple(sorted([tetrahedron[0], tetrahedron[1], tetrahedron[3]])))
        triangular_faces.add(tuple(sorted([tetrahedron[0], tetrahedron[2], tetrahedron[3]])))
        triangular_faces.add(tuple(sorted([tetrahedron[1], tetrahedron[2], tetrahedron[3]])))

    if DEBUG:
        print(f"Extracted {len(triangular_faces)} triangular faces.")
    return np.array(list(triangular_faces))

def visualize_point_cloud(point_cloud):
    """
    Visualizes a point cloud using Open3D's visualization tools.
    
    Parameters:
        point_cloud (open3d.geometry.PointCloud): Point cloud object to visualize.
    """
    try:
        if point_cloud is None or point_cloud.is_empty():
            raise ValueError("Invalid or empty point cloud. Cannot visualize.")
        if DEBUG:
            print("Visualizing point cloud...")
        #o3d.visualization.draw_geometries([point_cloud])
        o3d.io.write_point_cloud("./data/debug_outputs/output_point_cloud.ply", point_cloud)
        if DEBUG:
            print("Point cloud saved as output_point_cloud.ply")
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")

def visualize_triangulation(points, faces, output_image_path="./data/debug_outputs/triangulation_debug.png"):
    """
    Visualize the triangulated mesh and save it as an image.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Points')

    # Plot the faces
    for face in faces:
        vertices = points[face]
        poly = Poly3DCollection([vertices], alpha=0.5, edgecolor='r')
        ax.add_collection3d(poly)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Triangulation Debug Visualization")

    # Save the plot as an image
    plt.savefig(output_image_path)
    if DEBUG:
        print(f"Triangulation debug visualization saved to {output_image_path}")
    plt.close(fig)

def visualize_normals(point_cloud, output_image_path="./data/debug_outputs/normals_visualization.png"):
    """
    Visualize the point cloud with normals using Matplotlib and save the output to a file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2],
              length=0.1, normalize=True, color='r')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Point Cloud with Normals")

    # Save the visualization to a file
    output_image_path = os.path.abspath(output_image_path)
    plt.savefig(output_image_path)
    if DEBUG:
        print(f"Normals visualization saved to {output_image_path}")
    plt.close(fig)

def check_normals(point_cloud):
    """
    Check if normals are correctly estimated for the point cloud.
    """
    if not point_cloud.has_normals():
        raise ValueError("Point cloud does not have normals. Ensure normals are estimated before reconstruction.")
    if DEBUG:
        print("Normals are correctly estimated.")

def check_and_correct_face_normals(mesh):
    """
    Check and correct face normals to ensure all faces are oriented outwards.
    """
    if DEBUG:
        print("Checking and correcting face normals...")
    mesh.orient_triangles()
    mesh.compute_triangle_normals()
    if DEBUG:
        print("Face normals have been checked and corrected.")

def is_gpu_available():
    """
    Check if GPU acceleration is available for Open3D.

    Returns:
        bool: True if GPU is available, False otherwise.
    """
    try:
        import open3d.cuda.pybind as o3d_cuda
        return True
    except ImportError:
        return False