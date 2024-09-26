# MNIST Visualizer

This project is an interactive web-based application built using Dash, Plotly, and Bootstrap. It allows you to explore and visualize the MNIST dataset through various dimensionality reduction techniques, such as PCA, t-SNE, and UMAP. The app provides a user-friendly interface for visualizing latent spaces, individual images, and histograms of labels in the dataset.

## Features

- **Dimensionality Reduction**: Apply PCA, t-SNE, or UMAP to MNIST data and visualize the results in 2D or 3D.
- **Interactive Plotting**: Select the number of data points, latent dimensions, and reduction method with dropdowns.
- **Latent Space Visualization**: View the transformed space in 2D or 3D, with each point color-coded by its corresponding label.
- **Image Reconstruction**: Click on a point in the latent space to display the corresponding MNIST image.
- **Label Distribution Histogram**: View a histogram displaying the frequency of each MNIST label.
- **Responsive Layout**: The app is designed with Dash Bootstrap Components for a clean and responsive interface.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MNIST-Visualizer.git
   cd MNIST-Visualizer
   ```

2. Set up the environment (using conda or pip):
    ```
    conda create --name mnist-visualizer python=3.9
    conda activate mnist-visualizer
    pip install -r requirements.txt
    ```





    