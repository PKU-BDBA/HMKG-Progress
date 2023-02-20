from HMDB import download_HMDB,create_triples,create_subgraph_triples,convert_xml_to_json
from search import search,search_backward
from stats import summary,visualize_graph
from KGE import split_data,KGEmbedding


output_path=download_HMDB(url="https://hmdb.ca/system/downloads/current/saliva_metabolites.zip",file_path="data/saliva_metabolites.zip")

output_path=convert_xml_to_json(output_path,"data/saliva_metabolites.json")

triples=create_triples("/Users/colton/Desktop/代谢组学汇总/HMKG-Progress/HMKG/hmkg/data/saliva_metabolites.json")

split_data(triple_path=triples)

kge=KGEmbedding(model_name="TransE")

kge.construct_triples()
kge.save_id_mapping()
kge.KGE_model_pipeline()
