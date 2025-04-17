# Point Cloud to STL Tool
This tool converts point cloud data into an STL surface file.

## Installation

1. Install Python 3.10 or higher.
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the tool using the following command:
```bash
python pointCloudToSTL.py --input_file_path=<path_to_point_cloud> --algorithm=<algorithm>
```

Replace `<path_to_point_cloud>` with the path to your point cloud file and `<algorithm>` with one of `delaunay`, `poisson`, or `convex_hull`.

## Algorithm Suitability

### 1. Delaunay Triangulation
- **Best For**: Dense and evenly distributed point clouds.
- **Strengths**:
  - Produces a well-connected mesh.
  - Handles large datasets effectively.
- **Limitations**:
  - May produce inconsistent face normals, requiring post-processing.
  - Struggles with sparse or irregularly distributed points.
- **Use Case**: When the point cloud is dense and you need a quick triangulation.

### 2. Poisson Surface Reconstruction
- **Best For**: Smooth surfaces with well-defined normals.
- **Strengths**:
  - Produces smooth and watertight meshes.
  - Handles noise in the point cloud well.
- **Limitations**:
  - Requires accurate normals.
  - May struggle with sharp edges or fine details.
- **Use Case**: When the point cloud represents a smooth surface and normals are reliable.

### Poisson Algorithm Parameters

- **`poisson_depth`**: Specifies the depth of the octree used for surface reconstruction. Higher values result in more detailed meshes but require more memory and computation time. Default is `12`.
- **`density_percentile`**: Determines the density threshold for trimming the mesh. Vertices with densities below this percentile are removed. Default is `30`.

### 3. Convex Hull
- **Best For**: Simple geometries or when a quick outer boundary is needed.
- **Strengths**:
  - Guarantees a valid mesh with consistent face normals.
  - Computationally efficient for small datasets.
- **Limitations**:
  - Only captures the outer boundary, ignoring internal details.
  - Not suitable for complex or concave geometries.
- **Use Case**: When you need a quick approximation of the outer surface.
