
from .extract_from_external_database import get_CHEBI_cpd_info,get_KEGG_cpd_info,get_PubChem_cpd_info
from .extract_from_internal_database import get_taxonomy,get_property,get_concentrations,get_disease,get_reference,get_protein

from .utils import load_json
from tqdm import tqdm

def construct_KG(file_path,link_to_external_database=False,selected_metabolites=None):
    
    hmdb_data=load_json(file_path)

    entities=set()
    triples=set()
    for hmdb_id,meta_data in tqdm(hmdb_data.items()):
        if selected_metabolites is not None and hmdb_id not in selected_metabolites:
            continue
        
        hmdb_id="hmdb_id:"+hmdb_id
        entities.add((hmdb_id,"Hmdb_id"))
    
        try:
            entities.add(("chemical_formula:"+meta_data["chemical_formula"],"chemical_formula"))
            triples.add((hmdb_id,"chemical_formula","chemical_formula:"+meta_data["chemical_formula"]))
        
            entities.add(("average_molecular_weight:"+str(int(eval(meta_data["average_molecular_weight"]))),"average_molecular_weight"))
            triples.add((hmdb_id,"average_molecular_weight","average_molecular_weight:"+str(int(eval(meta_data["average_molecular_weight"])))))
        except:pass
    
        try:
            if meta_data["taxonomy"]:
                taxonomy_entity,taxonomy_triple=get_taxonomy(hmdb_id,meta_data["taxonomy"])
                entities.update(taxonomy_entity)
                triples.update(taxonomy_triple)
    
            if meta_data["experimental_properties"]:
                property_entity,property_triple=get_property(hmdb_id,meta_data["experimental_properties"],property_type="experimental")
                entities.update(property_entity)
                triples.update(property_triple)
        
            if meta_data["predicted_properties"]:
                property_entity,property_triple=get_property(hmdb_id,meta_data["predicted_properties"],property_type="predicted")
                entities.update(property_entity)
                triples.update(property_triple)
        
            if meta_data["biological_properties"]:
                property_entity,property_triple=get_property(hmdb_id,meta_data["biological_properties"],property_type="biological")
                entities.update(property_entity)
                triples.update(property_triple)
        
            if meta_data["normal_concentrations"]:
                concentration_entity,concentration_triple=get_concentrations(hmdb_id,meta_data["normal_concentrations"],concentration_type="normal")
                entities.update(concentration_entity)
                triples.update(concentration_triple)
        
            if meta_data["abnormal_concentrations"]:
                concentration_entity,concentration_triple=get_concentrations(hmdb_id,meta_data["abnormal_concentrations"],concentration_type="abnormal")
                entities.update(concentration_entity)
                triples.update(concentration_triple)
            
            if meta_data["diseases"]:
                disease_entity,disease_triple=get_disease(hmdb_id,meta_data["diseases"])
                entities.update(disease_entity)
                triples.update(disease_triple)
                
            if meta_data["general_references"]:
                reference_entity,reference_triple=get_reference(hmdb_id,meta_data["general_references"])
                entities.update(reference_entity)
                triples.update(reference_triple)
            
            if meta_data["protein_associations"]:
                protein_entity,protein_triple=get_protein(hmdb_id,meta_data["protein_associations"])
                entities.update(protein_entity)
                triples.update(protein_triple)
        except:pass
        
        if link_to_external_database:
            try:
                if meta_data["pubchem_compound_id"]!=None:
                    entities.add(("pubchem_id:"+meta_data["pubchem_compound_id"],"pubchem_id"))
                    triples.add((hmdb_id,"has_pubchem_id","pubchem_id:"+meta_data["pubchem_compound_id"]))
                    pubchem_entity,pubchem_triple=get_PubChem_cpd_info(meta_data["pubchem_compound_id"])
                    entities.update(pubchem_entity)
                    triples.update(pubchem_triple)
            except:pass
                
            try:
                if meta_data["kegg_id"]!=None:
                    entities.add(("kegg_id:"+meta_data["pubchem_compound_id"],"kegg_id"))
                    triples.add((hmdb_id,"has_kegg_id","kegg_id:"+meta_data["pubchem_compound_id"]))
                    kegg_entity,kegg_triple=get_KEGG_cpd_info(meta_data["kegg_id"])
                    entities.update(kegg_entity)
                    triples.update(kegg_triple)
            except:pass
                    
            try:
                if meta_data["chebi_id"]!=None:
                    entities.add(("chebi_id:"+meta_data["chebi_id"],"chebi_id"))
                    triples.add((hmdb_id,"has_chebi_id","chebi_id:"+meta_data["chebi_id"]))
                    chebi_id_entity,chebi_id_triple=get_CHEBI_cpd_info(meta_data["chebi_id"])
                    entities.update(chebi_id_entity)
                    triples.update(chebi_id_triple)
            except:pass

    return list(entities),list(triples)


def save_entity(entities):
    with open("data/entities.txt","w") as f:
        for entity in entities:
            f.write("\t".join(entity))
            f.write("\n")

def save_triple(triples):
    with open("data/triples.txt","w") as f:
        for triple in triples:
            f.write("\t".join(triple))
            f.write("\n")
