from .create_triples import create_triples

def create_subgraph_triples(input_path,hmdb_list):
    return create_triples(input_path=input_path,selected_metabolites=hmdb_list)