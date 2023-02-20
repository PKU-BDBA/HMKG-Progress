import csv
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
from collections import Counter


class HMKG():
    
    def __init__(self,model_name) -> None:
        self.model_name=model_name
        self.entity_path="data/Cancer400_entities.txt" # TODO 合并至KG生成过程中
    
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
    def create_triples(self, triple_path):
        """Reads a file containing triples and stores them as a list of tuples.

        Args:
            triple_path (str): The path of the file containing triples.

        Returns:
            list: A list of tuples, where each tuple contains three elements representing a triple.

        """
        with open(triple_path, "r", encoding="utf-8") as f:
            self.triples = f.readlines()

        self.triples = [i.strip("\n").split("\t") for i in tqdm(self.triples)]
        
        return self.triples
    
    
    @staticmethod
    def draw_statistics(counter, name, topk=20):
        """Draws a bar chart to visualize the top-k most frequent items in a counter.

        Args:
            counter (collections.Counter): The counter object containing the frequency count of items.
            name (str): The title of the plot.
            topk (int): The number of top-k items to plot. Default is 20.

        Returns:
            None.

        """
        counter_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:topk]
        keys = [k[:20] for k, v in counter_sorted]
        values = [v for k, v in counter_sorted]

        # Plot the bar chart
        plt.title(name)
        plt.xticks(rotation=45)
        plt.bar(keys, values)
        plt.show()

    
    # 数据统计
    def summary(self, show_bar_graph=True, save_result=False,topk=20):
        """Summarizes the statistics of the knowledge graph and saves them to a JSON file.

        Args:
            show_bar_graph (bool): Whether to display the bar graph of the frequency count. Default is True.
            topk (int): The number of top-k items to show in the bar graph. Default is 20.

        Returns:
            None.

        """
        statistics = {}

        G = nx.Graph()
        for h, r, t in tqdm(self.triples):
            G.add_edge(h, t, relation=r)

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        print("Number of nodes:", num_nodes)
        print("Number of edges:", num_edges)
        statistics["Nodes number"] = num_nodes
        statistics["Edges number"] = num_edges

        relation_counter = Counter([data['relation'] for u, v, data in G.edges(data=True)])
        head_counter = Counter([u for u, v, data in G.edges(data=True)])
        tail_counter = Counter([v for u, v, data in G.edges(data=True)])

        statistics[f"Top {topk} Relations"] = sorted(relation_counter.items(), key=lambda x: x[1], reverse=True)[:topk]
        statistics[f"Top {topk} Head Entities"] = sorted(head_counter.items(), key=lambda x: x[1], reverse=True)[:topk]
        statistics[f"Top {topk} Tail Entities"] = sorted(tail_counter.items(), key=lambda x: x[1], reverse=True)[:topk]

        if show_bar_graph:
            self.draw_statistics(relation_counter, "relations", topk)
            self.draw_statistics(head_counter, "heads", topk)
            self.draw_statistics(tail_counter, "tails", topk)

        if save_result:
            print("statistics results saved to results/statistics.json")
            with open("results/statistics.json", "w") as f:
                json.dump(statistics, f)

            
        
    def visualize_graph(self, node_num=20):
        """
        Visualize a random sample of the triples in the dataset as a graph.
        
        Args:
            node_num (int): The number of nodes to include in the graph.
            
        Returns:
            None
        """
        # Create a new graph with a random sample of the triples
        G = nx.Graph()
        for h, r, t in tqdm(random.sample(self.triples, node_num)):
            G.add_edge(h, t, relation=r)
            
        # Use spring layout to position the nodes and draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False)
        
        # Add edge labels to the graph
        labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        
        # Display the graph
        plt.show()
        

    # 信息查找
    def lookup(self, node_list, selected_relations, show_only=3,save_results=False,save_path="results/lookup.json"):
        """Lookup triplets based on specified nodes and relations and display the results.

        Args:
            node_list (list): List of nodes to include in search.
            selected_relations (list): List of relations to include in search.
            show_only (int, optional): Number of tail nodes to display. Defaults to 3.

        Returns:
            None
        """
        relation_count = {}

        for h, r, t in self.triples:
            if h in node_list and r in selected_relations:
                if r not in relation_count:
                    relation_count[r] = {}
                if t not in relation_count[r]:
                    relation_count[r][t] = 0
                relation_count[r][t] += 1

        lookup_result={}
        
        for r, tail_nodes in relation_count.items():
            if save_results:
                lookup_result[r]=tail_nodes
            sorted_tail_nodes = sorted(tail_nodes.items(), key=lambda x: x[1], reverse=True)
            print(f"Relation: {r}")
            for t, count in sorted_tail_nodes[:show_only]:
                print(f"  Tail Node: {t} Count: {count}")
        
        if save_results:
            print(f"lookup results saved to {save_path}")
            with open(save_path, "w") as f:
                json.dump(lookup_result, f)
    
    # 信息反查
    def lookup_backward(self, node_list, selected_relations, show_only=3,save_results=False,save_path="results/lookup_backward.json"):
        """Lookup triplets based on specified nodes and relations in backward direction and display the results.

        Args:
            node_list (list): List of nodes to include in search.
            selected_relations (list): List of relations to include in search.
            show_only (int, optional): Number of tail nodes to display. Defaults to 3.

        Returns:
            None
        """
        relation_count = {}

        for h, r, t in self.triples:
            if t in node_list and r in selected_relations:
                if r not in relation_count:
                    relation_count[r] = {}
                if h not in relation_count[r]:
                    relation_count[r][h] = 0
                relation_count[r][h] += 1

        lookup_result={}
        
        for r, head_nodes in relation_count.items():
            if save_results:
                lookup_result[r]=head_nodes
            sorted_head_nodes = sorted(head_nodes.items(), key=lambda x: x[1], reverse=True)
            print(f"Relation: {r}")
            for h, count in sorted_head_nodes[:show_only]:
                print(f"  Head Node: {h} Count: {count}")
        
        if save_results:
            print(f"lookup results saved to {save_path}")
            with open(save_path, "w") as f:
                json.dump(lookup_result, f)    

    # 链接Neo4j图数据库
    def Neo4j_converter(self):
        pass

    
    # Pykeen三元组构建
    def construct_triples(self,
                          train_path="data/TrainingSet.txt",
                          valid_path="data/EvaluationSet.txt",
                          test_path="data/TestSet.txt", 
                          create_inverse_triples=True):
        """Construct triples from the provided training, testing and validation sets.

        Args:
            train_path (str, optional): Path to training set file. Defaults to "data/TrainingSet.txt".
            valid_path (str, optional): Path to validation set file. Defaults to "data/EvaluationSet.txt".
            test_path (str, optional): Path to testing set file. Defaults to "data/TestSet.txt".
            create_inverse_triples (bool, optional): Whether or not to create inverse triples. Defaults to True.

        Returns:
            tuple: Tuple containing training, validation and testing data.
        """
        triple_factor_data = PathDataset(training_path=train_path,
                                         testing_path=test_path,
                                         validation_path=valid_path,
                                         create_inverse_triples=create_inverse_triples)
        triple_factor_data_train = triple_factor_data.training
        triple_factor_data_test = triple_factor_data.testing
        triple_factor_data_val = triple_factor_data.validation

        return triple_factor_data_train, triple_factor_data_val, triple_factor_data_test, triple_factor_data
    
    
    # 保存xx_to_id,id_to_xx文件
    def save_id_mapping(self, dir_path="data"):
        """Save id mapping of the constructed triples in JSON format to the specified directory.

        Args:
            dir_path (str, optional): Directory to save the files. Defaults to "data".

        Returns:
            None
        """
        _, _, _, triple_factor_data = self.construct_triples()

        if not os.path.exists(os.path.join(dir_path, self.model_name)):
            os.mkdir(os.path.join(dir_path, self.model_name))

        with open(os.path.join(dir_path, self.model_name, "entity_to_id.json"), "w") as f:
            json.dump(triple_factor_data.entity_to_id, f)
        with open(os.path.join(dir_path, self.model_name, "id_to_entity.json"), "w") as f:
            json.dump({i: j for j, i in triple_factor_data.entity_to_id.items()}, f)

        with open(os.path.join(dir_path, self.model_name, "relation_to_id.json"), "w") as f:
            json.dump(triple_factor_data.relation_to_id, f)
        with open(os.path.join(dir_path, self.model_name, "id_to_relation.json"), "w") as f:
            json.dump({i: j for j, i in triple_factor_data.relation_to_id.items()}, f)
        
    
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

        self.model = KGE_model(
            triples_factory=triple_factor_data_train,
            entity_representations=[pykeen.nn.Embedding],
            #entity_representations_kwargs=dict()
        )

        # Pick an optimizer from Torch
        optimizer = Adam(params=self.model.get_grad_params())

        # Pick a training approach (sLCWA or LCWA)
        training_loop = SLCWATrainingLoop(
            model=self.model,
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
            torch.save(self.model,f"checkpoints/{self.model_name}.pkl")

        return self.model


    # KGE测试
    def Evaluate_KGE(self,save_results=True):
        
        triple_factor_data_train,triple_factor_data_vld,triple_factor_data_tst,triple_factor_data=self.construct_triples()
        
        evaluator = RankBasedEvaluator()

        # Get triples to test
        mapped_triples = triple_factor_data_tst.mapped_triples
        
        # Evaluate
        results = evaluator.evaluate(
            model=self.model,
            mapped_triples=mapped_triples,
            batch_size=256,
            additional_filter_triples=[
                triple_factor_data_train.mapped_triples,
                triple_factor_data_vld.mapped_triples,
            ],
        )

        print(results.data)
        
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
    def save_all_embeddings(self):
        if not os.path.exists("embeddings/"):
            os.mkdir("embeddings")
        
        np.save(f"embeddings/{self.model_name}_Entity_Embedding.npy",self.model.entity_representations[0]._embeddings.weight.data.numpy())
        np.save(f"embeddings/{self.model_name}_Relation_Embedding.npy",self.model.relation_representations[0]._embeddings.weight.data.numpy())
            
    
    # 保存所有HMDB化合物的embedding
    def save_hmdb_embeddings(self):
        with open(f"data/{self.model_name}/entity_to_id.json","r") as f:
            entity_to_id=json.load(f)
        with open(f"data/{self.model_name}/id_to_entity.json","r") as f:
            id_to_entity=json.load(f)
        
        HMDBs=[i for i in entity_to_id.keys() if "HMDB" in i]
        HMDB_ids=[entity_to_id[i] for i in HMDBs]
    
        entity_embeddings=np.load(f"embeddings/{self.model_name}_Entity_Embedding.npy")
        HMDB_embedding_dict={}
        for i in HMDB_ids:
            HMDB_embedding_dict[id_to_entity[str(i)]]=entity_embeddings[i]
        
        with open(f"results/{self.model_name}_HMDB_Embedding.pkl","wb") as f:
            pickle.dump(HMDB_embedding_dict,f)

    
    # 根据输入的类别保存embedding
    def save_multiple_categories_embedding(self,categories):
        with open(f"data/{self.model_name}/entity_to_id.json","r") as f:
            entity_to_id=json.load(f)
        with open(f"data/{self.model_name}/id_to_entity.json","r") as f:
            id_to_entity=json.load(f)
        with open(self.entity_path, newline='', encoding='utf-8') as f:
            entity_list = f.readlines()
            entity_list = [i.strip("\r\n").split("\t") for i in entity_list]
        
        entities={}
        for entity in entity_list:
            if entity[1] in categories and entity[0] in entity_to_id.keys():
                if entity[1] not in entities.keys():
                    entities[entity[1]] = [entity_to_id[entity[0]]]
                else:
                    entities[entity[1]].append(entity_to_id[entity[0]])
        
        entity_embeddings=np.load(f"embeddings/{self.model_name}_Entity_Embedding.npy")
        
        entity_embeddings_dict={}
        for category in entities.keys():
            category_embedding={}
            for entity_id in entities[category]:
                    category_embedding[id_to_entity[str(entity_id)]]=entity_embeddings[entity_id]
            entity_embeddings_dict[category]=category_embedding
        
        with open(f"results/{self.model_name}_{'_'.join(categories)}_Embedding.pkl","wb") as f:
            pickle.dump(entity_embeddings_dict,f)


    # KGE pipeline
    def KGE_model_pipeline(self,eval_model=True,save_model=True,save_results=True,save_embeddings=True,save_HMDB_embedding=True,save_multiple_categories_embedding=None,save_id_mapping=True):
        
        if save_id_mapping:
            self.save_id_mapping(dir_path="data")
        
        self.Train_KGE(save_model=save_model)
        
        if save_embeddings:
            self.save_all_embeddings()
            
        if save_HMDB_embedding:
            self.save_hmdb_embeddings()
        
        if save_multiple_categories_embedding:
            self.save_multiple_categories_embedding(categories=save_multiple_categories_embedding)
            
        if eval_model:
            self.Evaluate_KGE(save_results=save_results)