# Point Cloud to STL Tool
This tool converts point cloud data into an STL surface file.

## Installation

1. Install Python 3.10 or higher.
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive CLI Menu
Run the tool with the `--cli-menu` flag to configure it interactively:
```bash
python pointCloudToSTL.py --cli-menu
```

### Command-Line Arguments
Run the tool with CLI flags for automation:
```bash
python pointCloudToSTL.py --input_file_path=<path_to_point_cloud> --algorithm=<algorithm> --output_stl_path=<output_path>
```

Replace `<path_to_point_cloud>` with the path to your point cloud file, `<algorithm>` with one of the supported algorithms, and `<output_path>` with the desired output STL file path.

### Help
Use the `--help` flag to see all available options:
```bash
python pointCloudToSTL.py --help
```

## Supported Algorithms

### 1. Delaunay Triangulation
- **Best For**: Dense and evenly distributed point clouds.
- **Strengths**:
  - Produces a well-connected mesh.
  - Handles large datasets effectively.
- **Limitations**:
  - May produce inconsistent face normals, requiring post-processing.
  - Struggles with sparse or irregularly distributed points.
- **Use Case**: When the point cloud is dense and you need a quick triangulation.

### 2. Poisson Surface Reconstruction (Recommended)
- **Best For**: Smooth surfaces with well-defined normals.
- **Strengths**:
  - Produces smooth and watertight meshes.
  - Handles noise in the point cloud well.
- **Limitations**:
  - Requires accurate normals.
  - May struggle with sharp edges or fine details.
- **Use Case**: When the point cloud represents a smooth surface and normals are reliable.

#### Poisson Algorithm Parameters
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

### 4. Ball Pivoting
- **Best For**: Dense and uniformly sampled point clouds.
- **Strengths**:
  - Produces high-quality meshes for dense datasets.
- **Limitations**:
  - Requires a well-sampled point cloud.
- **Use Case**: When the point cloud is dense and you need a detailed mesh.

### Experimental Algorithms
- **Marching Cubes**: Extracts surfaces from volumetric data.
- **Alpha Shapes**: Captures concave shapes with adjustable detail.
- **Radial Basis Function (RBF)**: Handles scattered and sparse data.
- **Voronoi-Based Reconstruction**: Constructs surfaces using Voronoi diagrams.
- **Moving Least Squares (MLS)**: Fits smooth surfaces to noisy data.

## Preprocessing and Postprocessing Features

### Preprocessing
- **Denoising**: Remove noise and outliers using statistical outlier removal.
- **Smoothing**: Smooth the point cloud using a Moving Least Squares (MLS) implementation.
- **Resampling**: Resample the point cloud to ensure uniform density using voxel downsampling.

### Postprocessing
- **Mesh Smoothing**: Apply Laplacian smoothing to remove jagged edges.
- **Hole Filling**: Close small gaps in the reconstructed mesh.
- **Mesh Simplification**: Reduce the number of triangles while preserving the overall shape.

## Performance Optimization
- **Large Point Cloud Support**: Downsample large point clouds to reduce memory usage and computation time.
- **Multi-Core Processing**: Use the `--max-cores` flag to control the number of CPU cores used.

## Example
```bash
python pointCloudToSTL.py --input_file_path=./data/inputData/example.xyz \
                          --algorithm=poisson \
                          --poisson_depth=10 \
                          --density_percentile=25 \
                          --enable_denoising=True \
                          --denoise_nb_neighbors=20 \
                          --denoise_std_ratio=1.5 \
                          --enable_simplification=True \
                          --simplify_target_reduction=0.5 \
                          --output_stl_path=./data/outputs/example_output.stl
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
