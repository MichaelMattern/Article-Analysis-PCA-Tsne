import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import base64

# Set the title of the app
st.title('PCA and t-SNE Visualization with Coloring')

# Sidebar for parameter inputs
st.sidebar.header('Parameters')

# Slider for selecting the number of PCA components
pca_dimensions = st.sidebar.slider(
    'Number of PCA Components', min_value=2, max_value=100, value=5, step=1
)

# Slider for selecting t-SNE perplexity
perplexity = st.sidebar.slider(
    't-SNE Perplexity', min_value=5, max_value=50, value=30, step=1
)

# Slider for selecting the number of samples to process
num_samples = st.sidebar.slider(
    'Number of Samples', min_value=100, max_value=30000, value=1000, step=100
)

# Slider for selecting the number of clusters
num_clusters = st.sidebar.slider(
    'Number of Clusters for Coloring', min_value=2, max_value=10, value=3, step=1
)

# File uploader for the dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Load the dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# Ensure the required columns are present
required_columns = [f'vector_{i}' for i in range(384)]
if not set(required_columns).issubset(df.columns):
    st.error("The dataset does not contain the required vector columns.")
    st.stop()

# Extract features and limit the number of samples
x = df.loc[0:num_samples - 1, required_columns].values

# Impute missing values
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Apply PCA
pca = PCA(n_components=pca_dimensions)
principalComponents = pca.fit_transform(x)
principal_component_names = [
    f'principal component {i + 1}' for i in range(pca_dimensions)
]
principalDf = pd.DataFrame(
    data=principalComponents, columns=principal_component_names
)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
data_2d = tsne.fit_transform(principalDf)

# Perform KMeans clustering for coloring
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(data_2d)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(
    data_2d[:, 0],
    data_2d[:, 1],
    c=labels,
    s=50,
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

# Download link for the t-SNE output
csv = tsne_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="tsne_output.csv">Download t-SNE CSV File with Clusters</a>'
st.markdown(href, unsafe_allow_html=True)
