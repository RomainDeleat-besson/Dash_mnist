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
   ```
   git clone git@github.com:RomainDeleat-besson/Dash_mnist.git
   cd Dash_mnist
   ```

2. Set up the environment 
    1. For conda users:
    ```
    conda create --name mnist-visualizer python=3.10
    conda activate mnist-visualizer
    pip install -r requirements.txt
    ```

    2. For pip users:
    ```
    python -m venv mnist-visualizer-env
    source mnist-visualizer-env/bin/activate
    pip install -r requirements.txt
    ```

3. Run the app:
    ```
    python MNIST_Visualizer.py
    ```

4. Access the app in your web browser:
    ```
    http://127.0.0.1:8050/
    ```

## Requirements

- Python 3.9 or higher
- Dash
- Dash Bootstrap Components
- Plotly
- numpy
- scikit-learn
- umap-learn

You can install all the dependencies via:
    ```
    pip install -r requirements.txt
    ```

## Usage

- Select the number of data points, reduction method (PCA, t-SNE, or UMAP), and the number of latent dimensions.
- Choose whether to display the latent space in 2D or 3D.
- Click on any point in the latent space to see the corresponding image.
- View the distribution of MNIST labels in the histogram on the right.

