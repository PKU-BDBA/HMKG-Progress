


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
                    entities.add(("pubmed_id:"+r["pubmed_id"],"Pubmed_id"))
                    triples.add(("biospecimen:"+_["biospecimen"],"has_reference","pubmed_id:"+r["pubmed_id"]))
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
                    entities.add(("pubmed_id:"+r["pubmed_id"],"Pubmed_id"))
                    triples.add(("disease:"+_["name"],"has_reference","pubmed_id:"+r["pubmed_id"]))
    except:pass
    
    return entities,triples


def get_reference(hmdb_id,reference_dict):
    entities=set()
    triples=set()
    
    try:
        for r in reference_dict["reference"]:
            if "pubmed_id" in r.keys() and r["pubmed_id"]:
                entities.add(("pubmed_id:"+r["pubmed_id"],"Pubmed_id"))
                triples.add((hmdb_id,"has_reference","reference:"+"pubmed_id"+r["pubmed_id"]))
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


def get_ontology(hmdb_id,ontology_dict):
    entities=set()
    triples=set()
    
    for i in ontology_dict["root"]:
        if type(i)==str:continue
        if i["term"]=="Physiological effect":
            if "descendants" not in i.keys():continue
            phy_descendants=i["descendants"]["descendant"]
            if type(phy_descendants)==dict:
                phy_descendants=[phy_descendants]
            for phy_descendants2 in phy_descendants:
                entities.add(("physiological_effect:"+phy_descendants2["term"],"Physiological_effect"))
                triples.add((hmdb_id,"has_physiological_effect","physiological_effect:"+phy_descendants2["term"]))
                if "descendants" not in phy_descendants2.keys():continue
                phy_descendants3=phy_descendants2["descendants"]["descendant"]
                get_synonym(entities,triples,phy_descendants2)
                if type(phy_descendants3)==dict:
                    phy_descendants=[phy_descendants3]
                else:
                    phy_descendants=phy_descendants3
                for phy_descendants3 in phy_descendants:
                    entities.add(("physiological_effect:"+phy_descendants3["term"],"Physiological_effect"))
                    triples.add((hmdb_id,"has_physiological_effect","physiological_effect:"+phy_descendants3["term"]))
                    triples.add(("physiological_effect:"+phy_descendants3["term"],"is_descendant_to","physiological_effect:"+phy_descendants2["term"]))
                    get_synonym(entities, triples, phy_descendants3)
                    if "descendants" not in phy_descendants3.keys():continue
                    phy_descendants4=phy_descendants3["descendants"]["descendant"]
                    if type(phy_descendants4)==dict:
                        phy_descendants=[phy_descendants4]
                    else:
                        phy_descendants=phy_descendants4
                    for phy_descendants4 in phy_descendants:
                        entities.add(("physiological_effect:"+phy_descendants4["term"],"Physiological_effect"))
                        triples.add((hmdb_id,"has_physiological_effect","physiological_effect:"+phy_descendants4["term"]))
                        triples.add(("physiological_effect:"+phy_descendants4["term"],"is_descendant_to","physiological_effect:"+phy_descendants3["term"]))
                        get_synonym(entities,triples,phy_descendants4)
                        if "descendants" not in phy_descendants4.keys():continue
                        phy_descendants5=phy_descendants4["descendants"]["descendant"]
                        if type(phy_descendants5)==dict:
                            phy_descendants=[phy_descendants5]
                        else:
                            phy_descendants=phy_descendants5
                        for phy_descendants5 in phy_descendants:
                            entities.add(("physiological_effect:"+phy_descendants5["term"],"Physiological_effect"))
                            triples.add((hmdb_id,"has_physiological_effect","physiological_effect:"+phy_descendants5["term"]))
                            triples.add(("physiological_effect:"+phy_descendants5["term"],"is_descendant_to","physiological_effect:"+phy_descendants4["term"]))
                            get_synonym(entities,triples,phy_descendants5)
    
        if i["term"]=="Disposition":
                if "descendants" not in i.keys():continue
                disposition=i["descendants"]["descendant"]
                if type(disposition)==dict:
                    disposition=[disposition]
                for disposition2 in disposition:
                    entities.add(("disposition:"+disposition2["term"],"Disposition"))
                    triples.add((hmdb_id,"has_disposition","disposition:"+disposition2["term"]))
                    if "descendants" not in disposition2.keys():continue
                    disposition3=disposition2["descendants"]["descendant"]
                    get_synonym(entities,triples,disposition2)
                    if type(disposition3)==dict:
                        disposition=[disposition3]
                    else:
                        disposition=disposition3
                    for disposition3 in disposition:
                        entities.add(("disposition:"+disposition3["term"],"Disposition"))
                        triples.add((hmdb_id,"has_disposition","disposition:"+disposition3["term"]))
                        triples.add(("disposition:"+disposition3["term"],"is_descendant_to","disposition:"+disposition2["term"]))
                        get_synonym(entities,triples,disposition3)
                        if "descendants" not in disposition3.keys():continue
                        disposition4=disposition3["descendants"]["descendant"]
                        if type(disposition4)==dict:
                            disposition=[disposition4]
                        else:
                            disposition=disposition4
                        for disposition4 in disposition:
                            entities.add(("disposition:"+disposition4["term"],"Disposition"))
                            triples.add((hmdb_id,"has_disposition","disposition:"+disposition4["term"]))
                            triples.add(("disposition:"+disposition4["term"],"is_descendant_to","disposition:"+disposition3["term"]))
                            get_synonym(entities,triples,disposition4)
                            if "descendants" not in disposition4.keys():continue
                            disposition5=disposition4["descendants"]["descendant"]
                            if type(disposition5)==dict:
                                disposition=[disposition5]
                            else:
                                disposition=disposition5
                            for disposition5 in disposition:
                                entities.add(("disposition:"+disposition5["term"],"Disposition"))
                                triples.add((hmdb_id,"has_disposition","disposition:"+disposition5["term"]))
                                triples.add(("disposition:"+disposition5["term"],"is_descendant_to","disposition:"+disposition4["term"]))
                                get_synonym(entities,triples,disposition5)
    
    
        if i["term"]=="Process":
                if "descendants" not in i.keys():continue
                process=i["descendants"]["descendant"]
                if type(process)==dict:
                    process=[process]
                for process2 in process:
                    entities.add(("process:"+process2["term"],"Process"))
                    triples.add((hmdb_id,"has_process","process:"+process2["term"]))
                    if "descendants" not in process2.keys():continue
                    process3=process2["descendants"]["descendant"]
                    get_synonym(entities,triples,process2)
                    if type(process3)==dict:
                        process=[process3]
                    else:
                        process=process3
                    for process3 in process:
                        entities.add(("process:"+process3["term"],"Process"))
                        triples.add((hmdb_id,"has_process","process:"+process3["term"]))
                        triples.add(("process:"+process3["term"],"is_descendant_to","process:"+process2["term"]))
                        get_synonym(entities,triples,process3)
                        if "descendants" not in process3.keys():continue
                        process4=process3["descendants"]["descendant"]
                        if type(process4)==dict:
                            process=[process4]
                        else:
                            process=process4
                        for process4 in process:
                            entities.add(("process:"+process4["term"],"Process"))
                            triples.add((hmdb_id,"has_process","process:"+process4["term"]))
                            triples.add(("process:"+process4["term"],"is_descendant_to","process:"+process3["term"]))
                            get_synonym(entities,triples,process4)
                            if "descendants" not in process4.keys():continue
                            process5=process4["descendants"]["descendant"]
                            if type(process5)==dict:
                                process=[process5]
                            else:
                                process=process5
                            for process5 in process:
                                entities.add(("process:"+process5["term"],"Process"))
                                triples.add((hmdb_id,"has_process","process:"+process5["term"]))
                                triples.add(("process:"+process5["term"],"is_descendant_to","process:"+process4["term"]))
                                get_synonym(entities,triples,process5)
    
    
    
        if i["term"]=="Role":
                if "descendants" not in i.keys():continue
                role=i["descendants"]["descendant"]
                if type(role)==dict:
                    role=[role]
                for role2 in role:
                    entities.add(("role:"+role2["term"],"Role"))
                    triples.add((hmdb_id,"has_role","role:"+role2["term"]))
                    if "descendants" not in role2.keys():continue
                    role3=role2["descendants"]["descendant"]
                    get_synonym(entities,triples,role2)
                    if type(role3)==dict:
                        role=[role3]
                    else:
                        role=role3
                    for role3 in role:
                        entities.add(("role:"+role3["term"],"Role"))
                        triples.add((hmdb_id,"has_role","role:"+role3["term"]))
                        triples.add(("role:"+role3["term"],"is_descendant_to","role:"+role2["term"]))
                        get_synonym(entities,triples,role3)
                        if "descendants" not in role3.keys():continue
                        role4=role3["descendants"]["descendant"]
                        if type(role4)==dict:
                            role=[role4]
                        else:
                            role=role4
                        for role4 in role:
                            entities.add(("role:"+role4["term"],"Role"))
                            triples.add((hmdb_id,"has_role","role:"+role4["term"]))
                            triples.add(("role:"+role4["term"],"is_descendant_to","role:"+role3["term"]))
                            get_synonym(entities,triples,role4)
                            if "descendants" not in role4.keys():continue
                            role5=role4["descendants"]["descendant"]
                            if type(role5)==dict:
                                role=[role5]
                            else:
                                role=role5
                            for role5 in role:
                                entities.add(("role:"+role5["term"],"Role"))
                                triples.add((hmdb_id,"has_role","role:"+role5["term"]))
                                triples.add(("role:"+role5["term"],"is_descendant_to","role:"+role4["term"]))
                                get_synonym(entities,triples,role5)
    
    return entities,triples

def get_synonym(entities, triples, dict_content):
    if "synonyms" in dict_content.keys():
        if dict_content["synonyms"]:
            if type(dict_content["synonyms"]["synonym"])==list:
                    for _ in dict_content["synonyms"]:
                        entities.add(("physiological_effect:"+_,"Physiological_effect"))
                        triples.add(("physiological_effect:"+_,"is_synonym_to","physiological_effect:"+dict_content["term"]))
            else:
                entities.add(("physiological_effect:"+dict_content["synonyms"]["synonym"],"Physiological_effect"))
                triples.add(("physiological_effect:"+dict_content["synonyms"]["synonym"],"is_synonym_to","physiological_effect:"+dict_content["term"]))
