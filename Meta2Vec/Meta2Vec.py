import pickle
import numpy as np
import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Meta2Vec():
    
    def __init__(self,embedding_path="embedding/TransE_HMDB_Embedding.pkl") -> None:
        self.load_embedding(embedding_path)
        self.get_available_hmdbs()
        
    
    def load_embedding(self,embedding_path="embedding/TransE_HMDB_Embedding.pkl"):
        self.embedding_path=embedding_path
        with open(self.embedding_path,"rb") as f:
            self.hmdb_embeddings=pickle.load(f)
        return self.hmdb_embeddings
    
    
    def get_available_hmdbs(self):
        self.vocab=list(self.hmdb_embeddings.keys())
        return self.vocab
    
    
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
        return list(similarity_dict.items())
    
    
    def most_similar(self,hmdb_1,topk=5):
        return self.cal_similarity_all(hmdb_1)[:topk]
    
    
    def visualize_umap(self):
        reducer = umap.UMAP()
        hmdb_2d_embedding = reducer.fit_transform(list(self.hmdb_embeddings.values()))
        sns.scatterplot(x=hmdb_2d_embedding[:,0], y=hmdb_2d_embedding[:,1])
        plt.show()
 
    
    
if __name__=="__main__":
    meta2vec=Meta2Vec()
    meta2vec.load_embedding()
    print(len(meta2vec.get_available_hmdbs()))
    meta2vec.get_hmdb_embedding(list(meta2vec.hmdb_embeddings.keys())[:10])
    print(meta2vec.cal_similarity("HMDB0000001","HMDB0000060"))
    meta2vec.visualize_umap()