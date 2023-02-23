


def get_taxonomy(hmdb_id,taxonomy_dict,keys_list=["direct_parent","kingdom","super_class","class","sub_class","molecular_framework","alternative_parents","substituents","external_descriptors"]):
    entities=set()
    triples=set()
    keys_list=list(set(keys_list)&set(taxonomy_dict.keys()))
    for key in keys_list:
        if key=="direct_parent":
            try:
                _=taxonomy_dict["direct_parent"]
                entities.add(("taxonomy_parent:"+_,"Taxonomy_Parent"))
                triples.add((hmdb_id,"has_direct_parent","taxonomy_parent:"+_))
            except:
                pass
        if key=="alternative_parents":
            try:
                _=taxonomy_dict["alternative_parents"]['alternative_parent']
                for i in _:
                    entities.add(("taxonomy_parent:"+i,"Taxonomy_Parent"))
                    triples.add((hmdb_id,"has_alternative_parent","taxonomy_parent:"+i))
            except:
                pass
        
        if key=="super_class":
            try:
                _=taxonomy_dict["super_class"]
                entities.add(("taxonomy_class:"+_,"Taxonomy_Class"))
                triples.add((hmdb_id,"has_super_class","taxonomy_class:"+_))
            except:
                pass
        if key=="class":
            try:
                _=taxonomy_dict["class"]
                entities.add(("taxonomy_class:"+_,"Taxonomy_Class"))
                triples.add((hmdb_id,"has_class","taxonomy_class:"+_))
            except:pass
        if key=="sub_class":
            try:
                _=taxonomy_dict["sub_class"]
                entities.add(("taxonomy_class:"+_,"Taxonomy_Class"))
                triples.add((hmdb_id,"has_sub_class","taxonomy_class:"+_))
            except:pass
            
        if key=="kingdom":
            try:
                _=taxonomy_dict["kingdom"]
                entities.add(("taxonomy_kingdom:"+_,"Taxonomy_Kingdom"))
                triples.add((hmdb_id,"has_kingdom","taxonomy_kingdom:"+_))
            except:pass
        if key=="molecular_framework":
            try:
                _=taxonomy_dict["molecular_framework"]
                entities.add(("taxonomy_molecular_framework:"+_,"Taxonomy_Molecular_Framework"))
                triples.add((hmdb_id,"has_molecular_framework","taxonomy_molecular_framework:"+_))
            except:
                pass
            
        if key=="substituents":
            try:
                _=taxonomy_dict["substituents"]['substituent']
                for i in _:
                    entities.add(("taxonomy_substituent:"+i,"Taxonomy_Substituent"))
                    triples.add((hmdb_id,"has_substituent","taxonomy_substituent:"+i))
            except:
                pass
        
        if key=="external_descriptors":
            try:
                _=taxonomy_dict["external_descriptors"]['external_descriptor']
                for i in _:
                    entities.add(("taxonomy_external_descriptor:"+i,"Taxonomy_External_Descriptor"))
                    triples.add((hmdb_id,"has_external_descriptor","taxonomy_external_descriptor:"+i))
            except:pass
    
    return entities,triples


def get_property(hmdb_id,property_dict,property_type="predicted",keys_list=["cellular_locations","biospecimen_locations","tissue_locations","pathways"]):
    entities=set()
    triples=set()
    
    if property_type in ["experimental","predicted"]:
        try:
            for property in property_dict["property"]:
                if property["value"] in ["Yes","No"]:
                    entities.add((property_type+"_property:"+property["kind"],"property"))
                    triples.add((hmdb_id,"has_"+property_type+"_property",property_type+"_property:"+property["kind"]))
        except:pass
        
    elif property_type=="biological":
        keys_list=list(set(keys_list)&set(property_dict.keys()))
        for key in keys_list:
            if key=="cellular_locations":
                try:
                    if type(property_dict["cellular_locations"]['cellular'])==str:
                        property_dict["cellular_locations"]['cellular']=[property_dict["cellular_locations"]['cellular']]
                    for _ in property_dict["cellular_locations"]['cellular']:
                        entities.add(("cellular_location:"+_,"Cellular_Location"))
                        triples.add((hmdb_id,"has_cellular_location","cellular_location:"+_))
                except:
                    pass
            if key=="biospecimen_locations":
                try:
                    for _ in property_dict["biospecimen_locations"]['biospecimen']:
                        entities.add(("biospecimen_location:"+_,"Biospecimen_Location"))
                        triples.add((hmdb_id,"has_biospecimen_location","biospecimen_location:"+_))
                except:pass
            if key=="tissue_locations":
                try:
                    if "tissue" in property_dict["tissue_locations"].keys():
                        for _ in property_dict["tissue_locations"]['tissue']:
                            entities.add(("tissue_location:"+_,"Tissue_Location"))
                            triples.add((hmdb_id,"has_tissue_location","tissue_location:"+_))
                except:pass
            if key=="pathways":
                try:
                    for _ in property_dict["pathways"]['pathway']:
                        if _["smpdb_id"]!=None:
                            entities.add(("pathway_id:"+_["smpdb_id"],"Pathway"))
                            triples.add((hmdb_id,"has_pathway","pathway_id:"+_["smpdb_id"]))
                        if _["kegg_map_id"]!=None:
                            entities.add(("pathway_id:"+_["kegg_map_id"],"Pathway"))
                            triples.add((hmdb_id,"has_pathway","pathway_id:"+_["kegg_map_id"]))
                except:pass
                    
    return entities,triples


def get_concentrations(hmdb_id,concentration_dict,concentration_type):
    entities=set()
    triples=set()
    
    try:
        for _ in concentration_dict["concentration"]:
            entities.add(("biospecimen:"+_["biospecimen"],"Biospecimen"))
            triples.add((hmdb_id,"has_"+concentration_type+"_concentration_in","biospecimen:"+_["biospecimen"]))
            if type(_["references"]["reference"])==dict:
                _["references"]["reference"]=[_["references"]["reference"]]
            for r in _["references"]["reference"]:
                if "pubmed_id" in r.keys() and r["pubmed_id"]:
                    entities.add(("reference:"+r["pubmed_id"],"Pubmed_id"))
                    triples.add(("biospecimen:"+_["biospecimen"],"has_reference","reference:"+r["pubmed_id"]))
    except:pass
    
    return entities,triples


def get_disease(hmdb_id,disease_dict):
    entities=set()
    triples=set()
    
    try:
        for _ in disease_dict["disease"]:
            entities.add(("disease:"+_["name"],"Disease"))
            triples.add((hmdb_id,"related_to_disease","disease:"+_["name"]))
            if _["omim_id"]:
                entities.add(("omim_id:"+_["omim_id"],"Omim_id"))
                entities.add(("disease:"+_["name"],"has_omim_id","omim_id:"+_["omim_id"]))
            if type(_["references"]["reference"])==dict:
                _["references"]["reference"]=[_["references"]["reference"]]
            for r in _["references"]["reference"]:
                if "pubmed_id" in r.keys() and r["pubmed_id"]:
                    entities.add(("reference:"+r["pubmed_id"],"Pubmed_id"))
                    triples.add(("disease:"+_["name"],"has_reference","reference:"+r["pubmed_id"]))
    except:pass
    
    return entities,triples


def get_reference(hmdb_id,reference_dict):
    entities=set()
    triples=set()
    
    try:
        for r in reference_dict["reference"]:
            if "pubmed_id" in r.keys() and r["pubmed_id"]:
                entities.add(("reference:"+r["pubmed_id"],"Pubmed_id"))
                triples.add((hmdb_id,"has_reference","reference:"+r["pubmed_id"]))
    except:pass
                
    return entities,triples


def get_protein(hmdb_id,protein_dict):
    entities=set()
    triples=set()
    
    try:
        for p in protein_dict["protein"]:
            if "protein_accession" in p.keys():
                entities.add(("hmdbp_id:"+p["protein_accession"],"Hmbdp_id"))
                triples.add((hmdb_id,"related_to","hmdbp_id:"+p["protein_accession"]))
            if "uniprot_id" in p.keys():
                entities.add(("uniprot_id:"+p["uniprot_id"],"Uniprot_id"))
                triples.add(("hmdbp_id:"+p["protein_accession"],"has_uniprot_id","uniprot_id:"+p["uniprot_id"]))
            if "gene_name" in p.keys():
                entities.add(("gene_name:"+p["gene_name"],"Gene_name"))
                triples.add(("hmdbp_id:"+p["protein_accession"],"related_to","gene_name:"+p["gene_name"]))
    except:pass
    
    return entities,triples