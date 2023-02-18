

def get_available_hmdbs(hmdb_embeddings):
    """
    Gets a list of all available HMDB IDs and assigns it to the hmdb_ids
    attribute.

    Returns:
    hmdb_ids (list): a list of all available HMDB IDs
    """
    hmdb_ids = list(hmdb_embeddings.keys())
    return hmdb_ids

def get_hmdb_embedding(hmdb_embeddings,hmdb_id_list):
    """
    Gets the embeddings of one or more HMDB IDs and returns them in a dictionary.

    Parameters:
    hmdb_id_list (str or list): a single HMDB ID as a string or a list of
    HMDB IDs

    Returns:
    hmdb_embedding (dict): a dictionary containing HMDB IDs as keys and
    their embeddings as values
    """
    if type(hmdb_id_list) == str:
        hmdb_id_list = [hmdb_id_list]
    hmdb_embedding = {}
    for id in hmdb_id_list:
        hmdb_embedding[id]=hmdb_embeddings[id]
    return hmdb_embedding
