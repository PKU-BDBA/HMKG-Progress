# HMKG

This package aims to download, preprocess, and analyze the Human Metabolome Database (HMDB) for metabolomics knowledge graph construction and knowledge graph embedding (KGE) tasks.

## Package Structure

The package is structured as follows:

* `HMKG.HMDB`: contains scripts to download, preprocess and convert HMDB data into triples and subgraphs.
* `HMKG.KGE`: contains scripts to construct triples and split data for KGE models.
* `HMKG.search`: contains scripts for searching the knowledge graph in both forward and backward directions.
* `HMKG.stats`: contains scripts to generate summary statistics, visualize the knowledge graph, and provide utility functions.

## Usage

To use this project, follow these steps:

1. Run `HMKG.HMDB.download_HMDB()` function from `HMDB/download.py` to download the HMDB dataset and save it to a local directory. This function takes two arguments: the URL to download the dataset and the file path to save it to.
2. Run `HMKG.HMDB.convert_xml_to_json()` function to convert the dataset from XML to JSON format. This function takes two arguments: the input file path (the downloaded HMDB dataset) and the output file path to save the converted data to.
3. Run `HMKG.HMDB.create_triples()` function to create triples from the converted dataset. This function takes one argument: the path to the converted HMDB dataset in JSON format.
4. Run `HMKG.KGE.split_data()` function to split the triples into training and testing datasets for KGE models. This function takes one argument: the path to the triple data.
5. Instantiate a KGE model from `HMKG.KGE.KGEmbedding` and run the `KGE_model_pipeline()` method to train the model. This script requires you to have already run `HMKG.KGE.split_data()` to create the training and testing datasets.
6. Use the scripts in `HMKG.search` to search the knowledge graph for entities and relationships.
7. Use the scripts in `HMKG.stats/` to generate summary statistics and visualize the knowledge graph.

## Dependencies

This project requires the following dependencies:

* Python 3
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* NetworkX

## Example Usage

```python
from HMKG.HMDB import download_HMDB, create_triples
from HMKG.search import search_backward
from HMKG.stats import summary
from HMKG.KGE import split_data, KGEmbedding

output_path = download_HMDB(url="https://hmdb.ca/system/downloads/current/saliva_metabolites.zip", file_path="data/saliva_metabolites.zip")

output_path = convert_xml_to_json(output_path, "data/saliva_metabolites.json")

triples = create_triples("data/saliva_metabolites.json")

split_data(triple_path=triples)

kge = KGEmbedding(model_name="TransE")

kge.construct_triples()
kge.save_id_mapping()
kge.KGE_model_pipeline()

```
