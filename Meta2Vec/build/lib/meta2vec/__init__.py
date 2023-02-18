import pickle
import os
import urllib
from distance import cal_list_similarity, hmdb_id_cosine_distance, embedding_cosine_distance, hmdb_id_euclidean_distance, embedding_euclidean_distance, most_similar
from utils import get_available_hmdbs, get_hmdb_embedding
from visualize import visualize_umap

def from_pretrained(model_type: str = "TransE", embedding_dir: str = "embedding") -> dict:
    """
    Downloads and loads the pretrained HMDB embeddings from the given model_type and returns them.

    Parameters:
    model_type (str): the type of the pretrained model (default: "TransE")
    embedding_dir (str): the directory where the embeddings will be saved (default: "embedding")

    Returns:
    hmdb_embeddings (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values
    """
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    embedding_path = os.path.join(embedding_dir, f"{model_type}_embedding.pkl")
    if not os.path.exists(embedding_path):
        urllib.request.urlretrieve(f"https://github.com/YuxingLu613/meta2vec/raw/main/embedding/{model_type}_HMDB_Embedding.pkl", embedding_path)
    return load_embedding_from_path(embedding_path)

def load_embedding_from_path(embedding_path: str = "embedding/TransE_HMDB_Embedding.pkl") -> dict:
    """
    Loads the HMDB embeddings from the given file path and returns them as a dictionary.

    Parameters:
    embedding_path (str): path to the HMDB embeddings file (default: "embedding/TransE_HMDB_Embedding.pkl")

    Returns:
    hmdb_embeddings (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values
    """
    with open(embedding_path, "rb") as f:
        hmdb_embeddings = pickle.load(f)
    return hmdb_embeddings
