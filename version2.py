import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import base64
from openTSNE import TSNE
import plotly.express as px

# Set the title of the app
st.title('PCA and t-SNE Visualization')

# Sidebar for parameter inputs
st.sidebar.header('Parameters')

# File uploader for the dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(uploaded_file)
    num_rows = df.shape[0]

    # Ensure the required columns are present
    required_columns = [f'vector_{i}' for i in range(384)]
    if not set(required_columns).issubset(df.columns):
        st.error("The dataset does not contain the required vector columns.")
        st.stop()

    # Slider for selecting the number of samples to process
    num_samples = st.sidebar.slider(
        'Number of Samples',
        min_value=100,
        max_value=num_rows,
        value=min(1000, num_rows),
        step=100
    )

    # Slider for selecting the number of PCA components
    pca_dimensions = st.sidebar.slider(
        'Number of PCA Components', min_value=2, max_value=50, value=5, step=1
    )

    # Slider for selecting t-SNE perplexity
    perplexity = st.sidebar.slider(
        't-SNE Perplexity', min_value=5, max_value=50, value=30, step=1
    )

    # Slider for selecting the number of clusters
    num_clusters = st.sidebar.slider(
        'Number of Clusters for Coloring', min_value=2, max_value=10, value=3, step=1
    )

    @st.cache_data
    def preprocess_data(df, num_samples, required_columns):
        # Randomly sample the data
        x = df.sample(n=num_samples, random_state=42)[required_columns].values
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        x = imputer.fit_transform(x)
        # Convert data type to float32 for efficiency
        x = x.astype('float32')
        return x

    x = preprocess_data(df, num_samples, required_columns)

    headlines = df.loc[0:num_samples - 1, 'Headlines'].values


    @st.cache_data
    def compute_pca(x, pca_dimensions):
        pca = PCA(n_components=pca_dimensions)
        principal_components = pca.fit_transform(x)
        return principal_components

    principal_components = compute_pca(x, pca_dimensions)

    @st.cache_data
    def compute_tsne(data, perplexity):
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_jobs=-1,  # Utilize all available CPU cores
            random_state=42
        )
        embedding = tsne.fit(data)
        return np.array(embedding)  # Convert embedding to NumPy array

    # Compute t-SNE on the PCA-reduced data
    data_2d = compute_tsne(principal_components, perplexity)

    # Perform KMeans clustering for coloring
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data_2d)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        data_2d[:, 0],
        data_2d[:, 1],
        c=labels,
        s=10,  # Reduced marker size for performance
        cmap='viridis',
        alpha=0.7
    )
    ax.set_xlabel('t-SNE Dimension 1', fontsize=15)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=15)
    ax.set_title('2D t-SNE Visualization with Clustering', fontsize=20)
    fig.colorbar(scatter, ax=ax, label='Cluster Label')

    st.pyplot(fig)

    # Create a DataFrame for t-SNE results
    tsne_df = pd.DataFrame(
        data_2d, columns=['tsne_dimension_1', 'tsne_dimension_2']
    )
    tsne_df['Cluster'] = labels
    tsne_df['Headlines'] = headlines

    # Download link for the t-SNE output
    csv = tsne_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="tsne_output.csv">'
        'Download t-SNE CSV File with Clusters</a>'
    )
    st.markdown(href, unsafe_allow_html=True)

    fig = px.scatter(
        tsne_df,
        x='tsne_dimension_1',
        y='tsne_dimension_2',
        color='Cluster',
        hover_data=['Headlines'],  # Include 'Headlines' for hover information
        title='2D t-SNE Visualization with Clustering',
        fontsize=20
    )
    # Customize axis labels, title, and colorbar
    fig.update_layout(
        xaxis_title='t-SNE Dimension 1',
        yaxis_title='t-SNE Dimension 2',
        title_font_size=20,
        coloraxis_colorbar=dict(
            title='Cluster Label'
        )
    )
    
    # Display the figure in Streamlit
    st.plotly_chart(fig)
else:
    st.warning("Please upload a CSV file.")
    st.stop()
