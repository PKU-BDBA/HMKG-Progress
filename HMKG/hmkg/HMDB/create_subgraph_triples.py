from .construct_graph import construct_KG

def create_subgraph_triples(input_path,hmdb_list,link_to_external_database=False,):
    return construct_KG(input_path=input_path,link_to_external_database=link_to_external_database,selected_metabolites=hmdb_list)