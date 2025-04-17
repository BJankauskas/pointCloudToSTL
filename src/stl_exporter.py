import numpy as np
import trimesh 

DEBUG= False  # Set to True to enable debug output

# Handles STL export functionality
def export_to_stl(mesh, output_path):
    pass

def save_mesh_as_stl(mesh, output_path):
    """
    Save the reconstructed mesh as an STL file in binary format using trimesh.
    """
    # Convert Open3D mesh to Trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Export as binary STL
    with open(output_path, 'wb') as f:
        f.write(trimesh_mesh.export(file_type='stl'))

    if DEBUG:
        print(f"Mesh saved to {output_path} in binary format")
