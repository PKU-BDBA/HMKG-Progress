import torch
import torchvision
import pickle
import pykeen.datasets
import json
from pykeen.datasets.base import PathDataset
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop,LCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
import os
import numpy as np
from pykeen.models import ConvE,TransE,TransD,TransH,TransR,KG2E,RotatE,DistMult
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class HMKG():
    
    def __init__(self,model_name) -> None:
        self.model_name=model_name
    
    # 下载HMDB
    def download_HMDB(self):
        pass
    
    
    # 格式转换
    def convert_xml_to_json(self,xml_file):
        pass

    
    # 构建HMKG子图
    def create_subgraph(self,hmdb_list):
        pass 
    
       
    # 构建三元组
    def creat_triples(self):
        with open("data/Cancer_Metabolism.txt","r") as f:
            self.triples=f.readlines()
        self.triples=[i.split("\t") for i in tqdm(self.triples)]
        return self.triples
    
    
    # 数据统计
    def graph_visualization_nx(self,show_pic=True):
        G = nx.Graph()
        for h, r, t in tqdm(random.sample(self.triples,30)):
            G.add_edge(h, t, relation=r)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False)
        labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        if show_pic:
            plt.show()
        

    # 信息查找
    def look_into(self,node_list,selected_relations,show_only=3):
        relation_count = {}

        for h, r, t in self.triples:
            if h in node_list and r in selected_relations:
                if r not in relation_count:
                    relation_count[r] = {}
                if t not in relation_count[r]:
                    relation_count[r][t] = 0
                relation_count[r][t] += 1

        for r, tail_nodes in relation_count.items():
            sorted_tail_nodes = sorted(tail_nodes.items(), key=lambda x: x[1], reverse=True)
            print(f"Relation: {r}")
            for t, count in sorted_tail_nodes[:show_only]:
                print(f"  Tail Node: {t} Count: {count}")

    
    # 信息反查
    def look_into_backward(self,node_list,selected_relations,show_only=3):
        relation_count = {}

        for h, r, t in self.triples:
            if t in node_list and r in selected_relations:
                if r not in relation_count:
                    relation_count[r] = {}
                if h not in relation_count[r]:
                    relation_count[r][h] = 0
                relation_count[r][h] += 1
        
        for r, tail_nodes in relation_count.items():
            sorted_tail_nodes = sorted(tail_nodes.items(), key=lambda x: x[1], reverse=True)
            print(f"Relation: {r}")
            for t, count in sorted_tail_nodes[:show_only]:
                print(f"  Tail Node: {t} Count: {count}")
    

    # 链接Neo4j图数据库
    def Neo4j_converter(self):
        pass

    
    # Pykeen三元组构建
    def construct_triples(self,
                          train_pth="data/TrainingSet.txt",
                          valid_pth="data/EvaluationSet.txt",
                          test_pth="data/TestSet.txt",
                          create_inverse_triples=True):
        
        triple_factor_data = PathDataset(training_path=train_pth, testing_path=test_pth, validation_path=valid_pth, create_inverse_triples=create_inverse_triples)
        triple_factor_data_train = triple_factor_data.training
        triple_factor_data_tst = triple_factor_data.testing
        triple_factor_data_vld = triple_factor_data.validation

        return triple_factor_data_train,triple_factor_data_vld,triple_factor_data_tst,triple_factor_data
    
    
    # 保存xx_to_id,id_to_xx文件
    def save_id_mapping(self,dir_path="data"):
        
        _,_,_,triple_factor_data=self.construct_triples()
        
        if not os.path.exists(os.path.join(dir_path,self.model_name)):
            os.mkdir(os.path.join(dir_path,self.model_name))
        
        with open(os.path.join(dir_path,self.model_name,"entity_to_id.json"),"w") as f:
            json.dump(triple_factor_data.entity_to_id,f)
        with open(os.path.join(dir_path,self.model_name,"id_to_entity.json"),"w") as f:
            json.dump({i:j for j,i in triple_factor_data.entity_to_id.items()},f)
        
        with open(os.path.join(dir_path,self.model_name,"relation_to_id.json"),"w") as f:
            json.dump(triple_factor_data.relation_to_id,f)
        with open(os.path.join(dir_path,self.model_name,"id_to_relation.json"),"w") as f:
            json.dump({i:j for j,i in triple_factor_data.relation_to_id.items()},f)
    
    
    # KGE训练
    def Train_KGE(self,save_model=True):
        
        if self.model_name=="ConvE":
            KGE_model=ConvE
        if self.model_name=="TransE":
            KGE_model=TransE
        if self.model_name=="TransD":
            KGE_model=TransD
        if self.model_name=="TransH":
            KGE_model=TransH
        if self.model_name=="TransR":
            KGE_model=TransR
        if self.model_name=="KG2E":
            KGE_model=KG2E
        if self.model_name=="RotatE":
            KGE_model=RotatE
        if self.model_name=="DistMult":
            KGE_model=DistMult

        triple_factor_data_train,triple_factor_data_vld,triple_factor_data_tst,triple_factor_data=self.construct_triples()
        
        print(triple_factor_data.summarize())

        model = KGE_model(
            triples_factory=triple_factor_data_train,
            entity_representations=[pykeen.nn.Embedding],
            #entity_representations_kwargs=dict()
        )

        # Pick an optimizer from Torch
        optimizer = Adam(params=model.get_grad_params())

        # Pick a training approach (sLCWA or LCWA)
        training_loop = SLCWATrainingLoop(
            model=model,
            triples_factory=triple_factor_data_train,
            optimizer=optimizer,
        )
        # Train
        _ = training_loop.train(
            triples_factory=triple_factor_data_train,
            num_epochs=5,
            batch_size=256,
        )
        
        if save_model:
            if not os.path.exists("checkpoints/"):
                os.mkdir("checkpoints")
            torch.save(model,f"checkpoints/{self.model_name}.pkl")

        return model


    # KGE测试
    def Evaluate_KGE(self,model,save_results=True):
        if type(model)==str:
            model=torch.load(model)
        
        
        triple_factor_data_train,triple_factor_data_vld,triple_factor_data_tst,triple_factor_data=self.construct_triples()
        
        evaluator = RankBasedEvaluator()

        # Get triples to test
        mapped_triples = triple_factor_data_tst.mapped_triples
        
        # Evaluate
        results = evaluator.evaluate(
            model=model,
            mapped_triples=mapped_triples,
            batch_size=256,
            additional_filter_triples=[
                triple_factor_data_train.mapped_triples,
                triple_factor_data_vld.mapped_triples,
            ],
        )

        print(results.data)
        print(results.metrics)
        
        if save_results:
            result_data_json = json.dumps({str(k): results.data[k] for k in results.data.keys()}, indent=4, ensure_ascii=False)

            if not os.path.exists("results/"):
                os.mkdir("results")
                
            write_res_pth = f"results/{self.model_name}_Cancer400.json"
            with open(write_res_pth, "w") as f:
                f.write(result_data_json)
            f.close()

        return results

        
    # 保存所有embedding
    def save_all_embeddings(self,model):
        if not os.path.exists("embeddings/"):
            os.mkdir("embeddings")
        
        if type(model)==str:
            model=torch.load(model)
        
        np.save(f"embeddings/{self.model_name}_Entity_Embedding.npy",model.entity_representations[0]._embeddings.weight.data.numpy())
        np.save(f"embeddings/{self.model_name}_Relation_Embedding.npy",model.relation_representations[0]._embeddings.weight.data.numpy())
            
    
    # 保存所有HMDB化合物的embedding
    def save_hmdb_embeddings(self,model):
        with open(f"data/{self.model_name}entity_to_id.json","r") as f:
            entity_to_id=json.load(f)
        with open(f"data/{self.model_name}id_to_entity.json","r") as f:
            id_to_entity=json.load(f)
        
        HMDBs=[i for i in entity_to_id.keys() if "HMDB" in i]
        HMDB_ids=[entity_to_id[i] for i in HMDBs]
    
        entity_embeddings=np.load(f"embeddings/{self.model_name}_Entity_Embedding.npy")
        HMDB_embedding_dict={}
        for i in HMDB_ids:
            HMDB_embedding_dict[id_to_entity[str(i)]]=entity_embeddings[i]
        
        with open(f"embeddings/{self.model_name}_HMDB_Embedding.pkl","wb") as f:
            pickle.dump(HMDB_embedding_dict,f)

    
    # 根据输入的类别保存embedding
    def save_multiple_categories_embedding(self,model,categories):
        pass


    # KGE pipeline
    def KGE_model_pipeline(self,eval_model=True,save_model=True,save_results=True,save_embeddings=True,save_HMDB_embedding=True,save_multiple_categories_embedding=None,save_id_mapping=True):
        
        if save_id_mapping:
            self.save_id_mapping(dir_path="data")
        
        model=self.Train_KGE(save_model=save_model)
        
        if save_embeddings:
            self.save_all_embeddings(model=model)
            
        if save_HMDB_embedding:
            self.save_hmdb_embeddings(model=model)
        
        if save_multiple_categories_embedding:
            self.save_multiple_categories_embedding(model=model,categories=None)
            
        if eval_model:
            self.Evaluate_KGE(model,save_results=save_results)