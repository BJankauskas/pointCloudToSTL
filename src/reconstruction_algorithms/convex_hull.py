import numpy as np
from scipy.spatial import ConvexHull

DEBUG = True

def extract_outer_surface(points):
    """
    Extract the outer surface of the point cloud using the convex hull algorithm.

    Parameters:
        points (numpy.ndarray): The input point cloud as a numpy array.

    Returns:
        numpy.ndarray: The indices of the vertices forming the convex hull faces.
    """
    if DEBUG:
        print("Extracting outer surface using convex hull...")
    hull = ConvexHull(points)
    faces = hull.simplices
    if DEBUG:
        print(f"Extracted {len(faces)} outer faces.")
    return faces
