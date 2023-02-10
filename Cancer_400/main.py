from Cancer400_KGE import HMKG


if __name__=="__main__":
    # hmkg=HMKG(model_name="TransE")
    # hmkg.save_id_mapping(dir_path="data")
    # model=hmkg.Train_KGE(save_model=True)
    # hmkg.save_all_embeddings(model=model)

    hmkg=HMKG(model_name="TransE")
    hmkg.KGE_model_pipeline(eval_model=True,
                            save_model=True,
                            save_results=True,
                            save_embeddings=True,
                            save_HMDB_embedding=True,
                            save_multiple_categories_embedding=True,
                            save_id_mapping=True)