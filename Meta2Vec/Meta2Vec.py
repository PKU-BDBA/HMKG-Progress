import pickle
import numpy as np


class Meta2Vec():
    
    
    def __init__(self,) -> None:
        return
    
    
    def load_embedding(self,embedding_path="embedding/TransE_HMDB_Embedding.pkl"):
        self.embedding_path=embedding_path
        with open(self.embedding_path,"rb") as f:
            self.hmdb_embeddings=pickle.load(f)
        return self.hmdb_embeddings
    
    
    def get_available_hmdbs(self):
        return list(self.hmdb_embeddings.keys())
    
    
    def get_hmdb_embedding(self,hmdb_id_list):
        if type(hmdb_id_list)==str:
            hmdb_id_list=[hmdb_id_list]
        hmdb_embedding={}
        for id in hmdb_id_list:
            hmdb_embedding.get(id,self.hmdb_embeddings[id])
        return hmdb_embedding
    
    
    def cal_similarity(self,hmdb_1,hmdb_2):
        hmdb_1_embedding=self.hmdb_embeddings[hmdb_1]
        hmdb_2_embedding=self.hmdb_embeddings[hmdb_2]
        return np.dot(hmdb_1_embedding, hmdb_2_embedding) / (np.linalg.norm(hmdb_1_embedding) * np.linalg.norm(hmdb_2_embedding))
    
    
    def cal_similarity_all(self,hmdb_1):
        hmdb_1_embedding=self.hmdb_embeddings[hmdb_1]
        similarity_dict={}
        for hmdb_id, embedding in self.hmdb_embeddings.items():
            if hmdb_id!=hmdb_1:
                similarity_dict[hmdb_id]=np.dot(hmdb_1_embedding, embedding) / (np.linalg.norm(hmdb_1_embedding) * np.linalg.norm(embedding))
        return similarity_dict
    
    
    
    
if __name__=="__main__":
    meta2vec=Meta2Vec()
    meta2vec.load_embedding()
    print(len(meta2vec.get_available_hmdbs()))
    meta2vec.get_hmdb_embedding(list(meta2vec.hmdb_embeddings.keys())[:10])
    print(meta2vec.cal_similarity("HMDB0000001","HMDB0000060"))