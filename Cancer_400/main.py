from HMKG import HMKG


if __name__=="__main__":
    # hmkg=HMKG(model_name="TransE")
    # hmkg.save_id_mapping(dir_path="data")
    # model=hmkg.Train_KGE(save_model=True)
    # hmkg.save_all_embeddings(model=model)

    hmkg=HMKG(model_name="TransE")
    hmkg.creat_triples()
    hmkg.graph_visualization_nx(show_graph=False,show_bar=True,topk=10)
    hmkg.look_into(node_list=["HMDB0000001","HMDB0000214","HMDB0005826","HMDB0008162"],selected_relations=["Disease"])
    hmkg.look_into_backward(node_list=["{'name': 'Obesity', 'omim_id': '601665'}\n","{'name': 'Pregnancy', 'omim_id': None}\n","{'name': 'Late-onset preeclampsia', 'omim_id': None}\n"],selected_relations=["Disease"])
    # hmkg.KGE_model_pipeline(eval_model=True,
    #                         save_model=True,
    #                         save_results=True,
    #                         save_embeddings=True,
    #                         save_HMDB_embedding=True,
    #                         save_multiple_categories_embedding=True,
    #                         save_id_mapping=True)