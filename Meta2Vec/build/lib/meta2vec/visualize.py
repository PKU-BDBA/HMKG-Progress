import umap
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_umap(hmdb_embeddings: dict):
    """
    Visualizes the UMAP embedding of HMDB IDs in 2D space.

    Parameters:
    hmdb_embeddings (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values

    Returns:
    None
    """
    reducer = umap.UMAP()
    hmdb_2d_embedding = reducer.fit_transform(list(hmdb_embeddings.values()))
    sns.scatterplot(x=hmdb_2d_embedding[:, 0], y=hmdb_2d_embedding[:, 1])
    plt.show()
