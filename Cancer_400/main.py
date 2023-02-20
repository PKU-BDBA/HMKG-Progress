from HMKG.hmkg import HMKG

if __name__=="__main__":

    hmkg=HMKG(model_name="TransE")
    hmkg.create_triples(triple_path="data/Cancer400_triples.txt")
    hmkg.summary(show_bar_graph=True,topk=10)
    hmkg.visualize_graph(node_num=30)
    hmkg.lookup(node_list=["HMDB0000001","HMDB0000214","HMDB0005826","HMDB0008162"],selected_relations=["Disease","synonym"],show_only=10,save_results=True)
    hmkg.lookup_backward(node_list=["{'name': 'Obesity', 'omim_id': '601665'}","{'name': 'Cystinuria', 'omim_id': '220100'}","{'name': 'Late-onset preeclampsia', 'omim_id': None}"],selected_relations=["Disease"],show_only=10,save_results=True)
    hmkg.KGE_model_pipeline(eval_model=True,
                            save_model=True,
                            save_results=True,
                            save_embeddings=True,
                            save_HMDB_embedding=True,
                            save_multiple_categories_embedding=["HMDB","Synonyms","Substituent"],
                            save_id_mapping=True)