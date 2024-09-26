import dash
import struct

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

from array import array
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP



def read_images_labels(images_filepath, labels_filepath):
    """
    Read images and labels from MNIST dataset
    """

    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img
    return np.array(images), np.array(labels)

def maker_colorbar_style(labels):
    colorscale = [
        [0, "rgb(68, 1, 84)"],        # Color for label 0
        [0.1, "rgb(68, 1, 84)"],        
        [0.1, "rgb(59, 82, 139)"],    # Color for label 1
        [0.2, "rgb(59, 82, 139)"],    
        [0.2, "rgb(33, 145, 140)"],   # Color for label 2
        [0.3, "rgb(33, 145, 140)"],
        [0.3, "rgb(94, 201, 98)"],    # Color for label 3
        [0.4, "rgb(94, 201, 98)"],
        [0.4, "rgb(253, 231, 37)"],   # Color for label 4
        [0.5, "rgb(253, 231, 37)"],
        [0.5, "rgb(255, 204, 0)"],    # Color for label 5
        [0.6, "rgb(255, 204, 0)"],
        [0.6, "rgb(240, 128, 128)"],  # Color for label 6
        [0.7, "rgb(240, 128, 128)"],
        [0.7, "rgb(150, 222, 173)"],  # Color for label 7
        [0.8, "rgb(150, 222, 173)"],
        [0.8, "rgb(133, 196, 186)"],  # Color for label 8
        [0.9, "rgb(133, 196, 186)"],
        [0.9, "rgb(81, 163, 252)"],   # Color for label 9
        [1, "rgb(81, 163, 252)"],
    ]
    marker_dict=dict(
        color=labels,  
        colorscale=colorscale,  
        showscale=True,
        colorbar=dict(
            tickvals=list(range(10)),
            ticktext=[str(i) for i in range(10)],
        ),
        size=8,
    )
    return marker_dict

def marker_histogram_style(unique_labels):
        # Define discrete colors for each label (0 to 9) using a color scale
        discrete_colors = [
            "rgb(68, 1, 84)",    # Label 0
            "rgb(59, 82, 139)",  # Label 1
            "rgb(33, 145, 140)", # Label 2
            "rgb(94, 201, 98)",  # Label 3
            "rgb(253, 231, 37)", # Label 4
            "rgb(255, 204, 0)",  # Label 5
            "rgb(240, 128, 128)",# Label 6
            "rgb(150, 222, 173)",# Label 7
            "rgb(133, 196, 186)",# Label 8
            "rgb(81, 163, 252)"  # Label 9
        ]
        # Assign a color for each unique label
        colors = [discrete_colors[label] for label in unique_labels]
        return colors



def main(app):
    # Load MNIST dataset
    path_img = "./data/10k-images-idx3-ubyte"
    path_lbl = "./data/10k-labels-idx1-ubyte"
    images_mnist, labels_mnist = read_images_labels(path_img, path_lbl)



    # Create the layout of the app
    app.layout = dbc.Container([
        # First Row: Dropdowns
        dbc.Row([
            dbc.Col(
                html.Div([
                    "Choose dimension of the plot: ",
                    dcc.Dropdown(
                        options=[
                            {"label": "2D plot", "value": "2D"}, 
                            {"label": "3D plot", "value": "3D"},
                         ],
                        value="2D",
                        id='dim_plot--dropdown',
                        clearable=False,
                    ),
                ]),
                width={"size": 2, "offset": 2},
            ),
            dbc.Col(
                html.Div([
                    "Choose latent dimensions: ",
                    dcc.Dropdown(
                        options=[
                            {"label": f"n_comp = {n_comp}", "value": n_comp} \
                                for n_comp in range(2,6)
                        ],
                        value=2,
                        id='n_comp--dropdown',
                        clearable=False,
                    ),
                ]),
                width={"size": 2},
            ),
            dbc.Col(
                html.Div([
                    "Number of points to display: ",
                    dcc.Dropdown(
                        options=[
                            {"label": f"nb point = {nb_points}", "value": nb_points} \
                                for nb_points in np.arange(100, 2100, 100, dtype=int)
                        ],
                        value=100,
                        id='nb_point--dropdown',
                        clearable=False,
                    ),
                ]),
                width={"size": 2},
            ),
            dbc.Col(
                html.Div([
                    "Choose method: ",
                    dcc.Dropdown(
                        options=[
                            {"label": "PCA",   "value": "PCA"}, 
                            {"label": "t-SNE", "value": "tsne"}, 
                            {"label": "UMAP",  "value": "UMAP"}
                        ],
                        value="PCA",
                        id='method--dropdown',
                        clearable=False,
                    ),
                ]),
                width={"size": 1},
            ),
        ]),
        
        # Second Row: Graphs
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id='latent_space--graph',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'latent_space'}}
                ),
                width=6,
            ),
            dbc.Col([
               dcc.Graph(
                    id='recons_img--graph',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'reconstructed_img'}},
                ),
                dcc.Graph(
                    id='Histo_label--graph',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'Histogram_label'}},
                ),
            ],
                width=6,
            )
        ]),

        # Third Row: Sliders
        dbc.Row([
            dbc.Col([
                html.Div([
                    "Dimension 1 :", 
                    dcc.Slider(
                        min=0,
                        max=4,
                        step=1,
                        id='dim1--slider',
                        value=0,
                    ),
                ]),
                html.Div([
                    "Dimension 2 :", 
                    dcc.Slider(
                        min=0,
                        max=4,
                        step=1,
                        id='dim2--slider',
                        value=1,
                    ),
                ]),
                html.Div([
                    "Dimension 3 :", 
                    dcc.Slider(
                        min=0,
                        max=4,
                        step=1,
                        id='dim3--slider',
                        value=2,
                    ),
                ]),
            ], 
            width=6,
            ),
        ]),

        dcc.Store(id="latent_space--store"),
        dcc.Store(id="data_img--store"),
        dcc.Store(id="data_lbl--store"),

    ], fluid=True)



    @app.callback(
        Output('data_img--store', 'data'),
        Output('data_lbl--store', 'data'),
        Input("nb_point--dropdown", "value"),
    )
    def choose_number_of_points_to_display(nb_point):
        """
        Function to update the stored MNIST images and labels based on the selected number of points.
        Parameters:
            nb_point (int): The number of MNIST data points to display, selected from a dropdown.
        Returns:
            tuple: A tuple containing two elements:
                - A list of MNIST images up to the selected number of points.
                - A list of MNIST labels corresponding to the selected images.
        """
        return images_mnist[:nb_point], labels_mnist[:nb_point]


    @app.callback(
        Output('latent_space--store', 'data'),
        Input('data_img--store', 'data'),
        Input("method--dropdown", "value"),
        Input("n_comp--dropdown", "value"),
    )
    def choose_reduction_method(images, method, n_comp):
        """
        Function to choose and apply a dimensionality reduction method on the input images.

        Parameters:
        -----------
        images : list
            List of images stored in the 'data_img--store' component.
        method : str
            The dimensionality reduction method selected from the 'method--dropdown' component.
            Options include 'PCA', 'tsne', and 'UMAP'.
        n_comp : int
            The number of components to keep, selected from the 'n_comp--dropdown' component.

        Returns:
        --------
        latent_space : np.ndarray
            The transformed latent space representation of the input images.
        """
        if method == 'PCA':  model = PCA(n_components=n_comp)
        if method == 'tsne': model = TSNE(n_components=n_comp, method='exact')
        if method == 'UMAP': model = UMAP(n_components=n_comp)
        images = np.array(images)
        latent_space = model.fit_transform(images.reshape(images.shape[0], images.shape[1]*images.shape[2]))
        return latent_space
    

    @app.callback(
        Output('latent_space--graph', 'figure'),
        Input('data_lbl--store', 'data'),
        Input('latent_space--store', 'data'),
        Input('dim_plot--dropdown', 'value'),
        Input('n_comp--dropdown', 'value'),
        Input('dim1--slider', 'value'),
        Input('dim2--slider', 'value'),
        Input('dim3--slider', 'value'),
    )
    def display_latent_space(labels, latent_space, dim, n_comp, dim1, dim2, dim3):
        """
        Function to display the latent space in either 2D or 3D.

        Parameters:
            labels (list): List of labels for the data points.
            latent_space (list): List of latent space coordinates.
            dim (str): Dimension of the plot, either "2D" or "3D".
            n_comp (int): Number of components in the latent space.
            dim1 (int): Index of the first dimension to plot.
            dim2 (int): Index of the second dimension to plot.
            dim3 (int): Index of the third dimension to plot (only used if dim is "3D").

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly figure object representing the latent space.
        """

        fig = go.Figure()
        latent_space = np.array(latent_space)
        if dim1 > n_comp-1: dim1 = n_comp-1
        if dim2 > n_comp-1: dim2 = n_comp-1
        if dim3 > n_comp-1: dim3 = n_comp-1

        if dim == "2D":
            plot_2D_latent_space = go.Scatter(
                x=latent_space[:, dim1], 
                y=latent_space[:, dim2], 
                mode='markers',
                marker=maker_colorbar_style(labels),
                text=labels,
            )
            fig.add_trace(plot_2D_latent_space)
            fig.update_layout(
                xaxis_title="dim "+str(dim1),
                yaxis_title="dim "+str(dim2),
            )

        if dim == "3D":
            if n_comp == 2: dim3 = 0
            plot_3D_latent_space = go.Scatter3d(
                x=latent_space[:, dim1], 
                y=latent_space[:, dim2], 
                z=latent_space[:, dim3],
                mode='markers',
                marker=maker_colorbar_style(labels),
                text=labels,
            )
            fig.add_trace(plot_3D_latent_space)
            fig.update_layout(
                scene = dict(
                    xaxis_title="dim "+str(dim1),
                    yaxis_title="dim "+str(dim2),
                    zaxis_title="dim "+str(dim3),
                ),
            )

        fig.update_layout(
            height=950,
            autosize=True,
            uirevision=True,
            template="plotly_white",
            margin=dict(l=10, r=10, b=10, t=50),
            title=f"Latent space: dim {dim1} - {dim2} - {dim3}",
        )
        return fig


    @app.callback(
        Output('recons_img--graph', 'figure'),
        Input('data_img--store', 'data'),
        Input('latent_space--graph', 'clickData'),
    )
    def display_clicked_image(images, clickData):
        """
        Function to display the clicked image from the latent space graph.
        Parameters:
        -----------
        images : list or np.ndarray
            List or array of images stored in the 'data_img--store'.
        clickData : dict
            Data from the click event on the 'latent_space--graph'. Contains information about the clicked point.
        Returns:
        --------
        plotly.graph_objs.Figure or dash.no_update
            A Plotly figure object containing the clicked image formatted for display, or no update if no point was clicked.
        """

        if clickData is None: return dash.no_update # Don't update if no point was clicked
        
        # Get the index of the clicked point
        clicked_index = clickData['points'][0]['pointNumber']
        
        # Tricks to display the image in the right format
        # (Only with go.Image)
        img = images[clicked_index]  
        img = np.stack([img, img, img, img], axis=-1)

        fig = go.Figure()
        fig.add_trace(go.Image(z=img, colormodel='rgb'))
        fig.update_layout(
            height=500,
            autosize=True,
            uirevision=True,
            title="Clicked Image",
            margin=dict(l=5, r=5, b=5, t=50),
            template="plotly_white",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig


    @app.callback(

        Output("Histo_label--graph", "figure"),
        Input('data_lbl--store', 'data'),
    )
    def display_histogram_of_mnist_number(labels):
        """
        Callback function to display a histogram of MNIST labels.

        This function is triggered by changes in the 'data_lbl--store' component.
        It calculates the frequency of each label in the provided dataset and 
        generates a bar chart to visualize the distribution of MNIST labels.

        Args:
            labels (list or array-like): The list or array of MNIST labels.

        Returns:
            plotly.graph_objs._figure.Figure: A Plotly Figure object containing 
            the histogram of MNIST labels.
        """

        unique_labels, counts = np.unique(labels, return_counts=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=unique_labels,
            y=counts,
            marker=dict(color=marker_histogram_style(unique_labels)),
        ))

        fig.update_layout(
            height=450,
            autosize=True,
            template="plotly_white",
            title="Distribution of MNIST Labels",
            xaxis=dict(
                title="Label",
                tickmode="linear",
                dtick=1,
            ),
            margin=dict(l=5, r=5, b=5, t=50),
            yaxis=dict(title="Count")
        )
        return fig



if __name__ == "__main__":
    # Create the Dash app with a Bootstrap theme
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Run the main function
    main(app)

    # Run the Dash app
    app.run_server(debug=True)

































