from HMDB import download_HMDB,create_triples,create_subgraph_triples,convert_xml_to_json,construct_graph,save_entity,save_triple
from search import search,search_backward
from stats import summary,visualize_graph
from KGE import split_data,KGEmbedding


# output_path=download_HMDB(url="https://hmdb.ca/system/downloads/current/saliva_metabolites.zip",file_path="data/saliva_metabolites.zip")

# output_path=convert_xml_to_json(output_path,"data/saliva_metabolites.json")

entities,triples=construct_graph.construct_KG("/Users/colton/Desktop/代谢组学汇总/other databases/part_of_hmdb.json",link_to_external_database=False)

split_data(triple_path=triples)

kge=KGEmbedding(model_name="TransE")

kge.construct_triples()
kge.save_id_mapping()
kge.KGE_model_pipeline()
