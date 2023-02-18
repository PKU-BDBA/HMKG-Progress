import numpy as np

def hmdb_id_cosine_distance(hmdb_embeddings: dict, hmdb_1: str, hmdb_2: str) -> float:
    """
    Calculates the cosine distance between two HMDB IDs using their embeddings.

    Parameters:
    hmdb_embeddings (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values
    hmdb_1 (str): the first HMDB ID to compare
    hmdb_2 (str): the second HMDB ID to compare

    Returns:
    distance (float): the cosine distance between the two HMDB IDs
    """
    hmdb_1_embedding = hmdb_embeddings[hmdb_1]
    hmdb_2_embedding = hmdb_embeddings[hmdb_2]
    distance = np.dot(hmdb_1_embedding, hmdb_2_embedding) / (np.linalg.norm(hmdb_1_embedding) * np.linalg.norm(hmdb_2_embedding))
    return distance

def hmdb_id_euclidean_distance(hmdb_embeddings: dict, hmdb_1: str, hmdb_2: str) -> float:
    """
    Calculates the Euclidean distance between two HMDB IDs using their embeddings.

    Parameters:
    hmdb_embeddings (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values
    hmdb_1 (str): the first HMDB ID to compare
    hmdb_2 (str): the second HMDB ID to compare

    Returns:
    distance (float): the Euclidean distance between the two HMDB IDs
    """
    hmdb_1_embedding = hmdb_embeddings[hmdb_1]
    hmdb_2_embedding = hmdb_embeddings[hmdb_2]
    distance=np.sqrt(np.sum((hmdb_1_embedding - hmdb_2_embedding)**2))
    return distance

def embedding_cosine_distance(hmdb_1_embedding: np.ndarray, hmdb_2_embedding: np.ndarray) -> float:
    """
    Calculates the cosine distance between two embeddings.

    Parameters:
    hmdb_1_embedding (ndarray): an embedding vector
    hmdb_2_embedding (ndarray): another embedding vector

    Returns:
    distance (float): the cosine distance between the two embedding vectors
    """
    distance = np.dot(hmdb_1_embedding, hmdb_2_embedding) / (np.linalg.norm(hmdb_1_embedding) * np.linalg.norm(hmdb_2_embedding))
    return distance

def embedding_euclidean_distance(hmdb_1_embedding: np.ndarray, hmdb_2_embedding: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two embeddings.

    Parameters:
    hmdb_1_embedding (ndarray): an embedding vector
    hmdb_2_embedding (ndarray): another embedding vector

    Returns:
    distance (float): the Euclidean distance between the two embedding vectors
    """
    distance=np.sqrt(np.sum((hmdb_1_embedding - hmdb_2_embedding)**2))
    return distance

def cal_list_similarity(hmdb_embeddings, hmdb_1, hmdb_list, distance_type="cosine"):
    """
    Calculates the similarity between an HMDB compound and a list of other
    HMDB compounds.

    Parameters:
    hmdb_embeddings (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values
    hmdb_1 (str): the HMDB ID of the first compound
    hmdb_list (list): a list of HMDB IDs to compare to the first compound
    distance_type (str): the type of distance metric to use, either "cosine"
    or "euclidean"

    Returns:
    similarity_list (list): a list of tuples, where each tuple contains an
    HMDB ID and its similarity score with the first compound
    """
    similarity_dict = {}
    for hmdb_id in hmdb_list:
        if hmdb_id != hmdb_1:
            if distance_type == "cosine":
                similarity_dict[hmdb_id] = hmdb_id_cosine_distance(hmdb_embeddings, hmdb_1, hmdb_id)
            if distance_type == "euclidean":
                similarity_dict[hmdb_id] = hmdb_id_euclidean_distance(hmdb_embeddings, hmdb_1, hmdb_id)
    similarity_list = list(similarity_dict.items())
    return similarity_list
    
def most_similar(hmdb_embeddings, hmdb_1, hmdb_list, topk=5, distance_type="cosine"):
    """
    Returns the top k most similar HMDB compounds to a given compound.

    Parameters:
    hmdb_embeddings (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values
    hmdb_1 (str): the HMDB ID of the first compound
    hmdb_list (list): a list of HMDB IDs to compare to the first compound
    topk (int): the number of top similar compounds to return
    distance_type (str): the type of distance metric to use, either "cosine"
    or "euclidean"

    Returns:
    topk_similar (list): a list of tuples, where each tuple contains an
    HMDB ID and its similarity score with the first compound, sorted in
    descending order of similarity score
    """
    similarity_list = cal_list_similarity(hmdb_embeddings, hmdb_1, hmdb_list, distance_type)
    topk_similar = sorted(similarity_list, key=lambda x: x[1], reverse=True)[:topk]
    return topk_similar
