from HMDB import download_HMDB,create_triples,create_subgraph_triples,convert_xml_to_json,construct_graph,save_entity,save_triple,load_entity,load_triple
from search import search,search_backward
from stats import summary,visualize_graph
from KGE import split_data,KGEmbedding
import csv


# output_path=download_HMDB(url="https://hmdb.ca/system/downloads/current/saliva_metabolites.zip",file_path="data/saliva_metabolites.zip")

# output_path=convert_xml_to_json(output_path,"data/saliva_metabolites.json")

# entities,triples=construct_graph.construct_KG("/home/luyx/HMKG/data/hmdb_metabolities.json",link_to_external_database=False)

# save_entity(entities)
# save_triple(triples)

# entities=load_entity()
# triples=load_triple()

# summary(triples,show_bar_graph=False,save_result=True,topk=100)

# split_data(triple_path=triples)

kge=KGEmbedding(model_name="TransE")

kge.construct_triples()
kge.save_id_mapping()
kge.KGE_model_pipeline()

# TransE  ('hits_at_10', 'head', 'optimistic'): 0.15977071391349662,
# TransH  ('hits_at_10', 'head', 'optimistic'): 0.6548202188639917,