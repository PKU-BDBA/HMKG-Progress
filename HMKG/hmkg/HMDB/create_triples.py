import csv
from .utils import clean_quote,load_json,drop_duplicate
from tqdm import tqdm


def create_triples(input_path,selected_metabolites=None):
    
    json_data = load_json(input_path)["hmdb"]["metabolite"]
    
    HMDB_list = []
    Name_Synonyms = []
    Taxonomy_list = []
    Alternative_parent = []
    Substituent_list = []
    External_descriptor = []
    Ontology_list = []
    External_property_list = []
    Predicted_property_list = []
    Spectrum_list = []
    Tissue_list = []
    Pathway_list = []
    Reference_list = []
    Disease_list = []
    Protein_list = []
    HMDB_dict_list = []
    Cellular_locations = []
    Biospecimen_locations = []
    Concentration = []
    Protein_list = []

    Rel_secondary_accession = []
    Rel_synonyms = []
    Rel_alternative_parent = []
    Rel_substituent = []
    Rel_external_descriptor = []
    Rel_taxonomy = []
    Rel_external_property = []
    Rel_predicted_property = []
    Rel_spectrum = []
    Rel_cellular_location = []
    Rel_biospecimen_location = []
    Rel_tissue = []
    Rel_pathway = []
    Rel_concentration = []
    Rel_reference_concentration = []
    Rel_reference_disease = []
    Rel_disease = []
    Rel_general_reference = []
    Rel_protein = []


    ## tqdm 用于添加进度条
    for h_value in tqdm(json_data):
        h_key=h_value["accession"]
        if selected_metabolites is not None and h_key not in selected_metabolites:
            continue
        id = 0
        HMDB_NO = h_key
        HMDB_dict = {}
        for key, value in h_value.items():

            if not value:
                continue

            if key == "secondary_accessions":

                if isinstance(value["accession"], str):
                    HMDB_list.append(clean_quote(value["accession"]))
                    Rel_secondary_accession.append(
                        [HMDB_NO, "secondary_accession", clean_quote(value["accession"])])
                else:
                    for v in value["accession"]:
                        HMDB_list.append(clean_quote(v))
                        Rel_secondary_accession.append(
                            [HMDB_NO, "secondary_accession", clean_quote(v)])

            elif key == 'synonyms':
                if isinstance(value["synonym"], str):
                    Name_Synonyms.append(clean_quote(value["synonym"]))
                    Rel_synonyms.append(
                        [HMDB_NO, "synonym", clean_quote(value["synonym"])])
                else:
                    for v in value["synonym"]:
                        Name_Synonyms.append(clean_quote(v))
                        Rel_synonyms.append([HMDB_NO, "synonym", clean_quote(v)])

            elif key == "taxonomy":
                if "alternative_parents" in value.keys():
                    alternative_parent = value.pop("alternative_parents")
                    if alternative_parent:
                        for v in list(alternative_parent.values())[0]:
                            Alternative_parent.append(clean_quote(v))
                            Rel_alternative_parent.append(
                                [HMDB_NO, "alternative_patient", clean_quote(v)])

                if "substituents" in value.keys():
                    substituent = value.pop("substituents")
                    if substituent:
                        for v in list(substituent.values())[0]:
                            Substituent_list.append(clean_quote(v))
                            Rel_substituent.append(
                                [HMDB_NO, "substituent", clean_quote(v)])
                if "external_descriptors" in value.keys():
                    external_descriptor = value.pop("external_descriptors")
                    if external_descriptor:
                        for v in list(external_descriptor.values())[0]:
                            External_descriptor.append(clean_quote(v))
                            Rel_external_descriptor.append(
                                [HMDB_NO, "external_descriptor", clean_quote(v)])
                value["HMDB_NO"] = HMDB_NO
                Taxonomy_list.append(value)
                Rel_taxonomy.append([HMDB_NO, "taxonomy", value])

            elif key == "experimental_properties":
                if isinstance(value["property"], dict):
                    External_property_list.append(value["property"])
                    Rel_external_property.append(
                        [HMDB_NO, "experimental_properties", value["property"]])
                else:
                    for v in value["property"]:
                        External_property_list.append(v)
                        Rel_external_property.append(
                            [HMDB_NO, "experimental_properties", v])

            elif key == "predicted_properties":
                if isinstance(value["property"], dict):
                    Predicted_property_list.append(value["property"])
                    Rel_predicted_property.append(
                        [HMDB_NO, "predicted_properties", value["property"]])
                else:
                    for v in value["property"]:
                        v["value"] = v["value"].replace("(", "")
                        v["value"] = v["value"].replace(")", "")
                        Predicted_property_list.append(v)
                        Rel_predicted_property.append(
                            [HMDB_NO, "predicted_properties", v])

            elif key == "spectra":
                if value:
                    if isinstance(value["spectrum"], dict):
                        Spectrum_list.append(value["spectrum"])
                        Rel_spectrum.append(
                            [HMDB_NO, "spectrum", value["spectrum"]])
                    else:
                        for v in value["spectrum"]:
                            Spectrum_list.append(v)
                            Rel_spectrum.append([HMDB_NO, "spectrum", v])

            elif key == "biological_properties":
                cellular_locations = value.pop("cellular_locations")
                if cellular_locations:
                    for v in list(cellular_locations.values()):
                        if isinstance(v, str):
                            Cellular_locations.append(clean_quote(v))
                            Rel_cellular_location.append(
                                [HMDB_NO, "cellular_locations", clean_quote(v)])
                        else:
                            for _ in v:
                                Cellular_locations.append(clean_quote(_))
                                Rel_cellular_location.append(
                                    [HMDB_NO, "cellular_locations", clean_quote(_)])
                biospecimen_locations = value.pop("biospecimen_locations")
                if biospecimen_locations:
                    for v in list(biospecimen_locations['biospecimen']):
                        Biospecimen_locations.append(clean_quote(v))
                        Rel_biospecimen_location.append(
                            [HMDB_NO, "biospecimen", clean_quote(v)])
                tissue_locations = value.pop("tissue_locations")
                if tissue_locations:
                    for v in list(tissue_locations):
                        Tissue_list.append(v)
                        Rel_tissue.append([HMDB_NO, "tissue", clean_quote(v)])
                pathways = value.pop("pathways")
                if pathways:
                    for v in list(pathways["pathway"]):
                        if not isinstance(v, dict):
                            continue
                        Pathway_list.append(v)
                        Rel_pathway.append([HMDB_NO, "pathway", v])

            elif key == "normal_concentrations":
                for v in value['concentration']:
                    if isinstance(v, str):
                        continue
                    else:
                        if "references" in v.keys():
                            reference = v.pop("references")
                            if reference and isinstance(reference["reference"], list):
                                for i in reference["reference"]:
                                    i["reference_text"] = clean_quote(
                                        i["reference_text"])
                                    Reference_list.append(i)
                                    Rel_reference_concentration.append(
                                        [v, "reference", i])
                            elif reference and isinstance(reference["reference"], dict):
                                reference["reference"]["reference_text"] = clean_quote(
                                    reference["reference"]["reference_text"])
                                Reference_list.append(reference["reference"])
                                Rel_reference_concentration.append(
                                    [v, "reference", reference["reference"]])
                    v["id"] = str(id)
                    v["status"] = "Normal"
                    id += 1
                    Concentration.append(v)
                    Rel_concentration.append([HMDB_NO, "Concentration", v])

            elif key == "abnormal_concentrations":
                for v in value['concentration']:
                    if isinstance(v, str):
                        continue
                    else:
                        if "references" in v.keys():
                            reference = v.pop("references")
                            if reference and isinstance(reference["reference"], list):
                                for i in reference["reference"]:
                                    i["reference_text"] = clean_quote(
                                        i["reference_text"])
                                    Reference_list.append(i)
                                    Rel_reference_concentration.append(
                                        [v, "reference", i])
                            elif reference and isinstance(reference["reference"], dict):
                                reference["reference"]["reference_text"] = clean_quote(
                                    reference["reference"]["reference_text"])
                                Reference_list.append(reference["reference"])
                                Rel_reference_concentration.append(
                                    [v, "reference", reference["reference"]])
                            v["id"] = str(id)
                            v["status"] = "Abnormal"
                            id += 1
                            Concentration.append(v)
                            Rel_concentration.append([HMDB_NO, "Concentration", v])

            elif key == "diseases":
                for v in value["disease"]:
                    if isinstance(v, str):
                        continue
                    else:
                        if "references" in v.keys():
                            reference = v.pop("references")
                            if isinstance(reference["reference"], list):
                                for i in reference["reference"]:
                                    i["reference_text"] = clean_quote(
                                        i["reference_text"])
                                    Reference_list.append(i)
                                    Rel_reference_disease.append(
                                        [v, "reference", i])
                            elif isinstance(reference["reference"], dict):
                                reference["reference"]["reference_text"] = clean_quote(
                                    reference["reference"]["reference_text"])
                                Reference_list.append(reference["reference"])
                                Rel_reference_disease.append(
                                    [v, "reference", reference["reference"]])
                            v["name"] = clean_quote(v["name"])
                            Disease_list.append(v)
                            Rel_disease.append([HMDB_NO, "Disease", v])

            elif key == "general_references":
                for v in value["reference"]:
                    if isinstance(v, str):
                        continue
                    else:
                        Reference_list.append(v)
                        Rel_general_reference.append([HMDB_NO, "reference", v])

            elif key == "protein_associations":
                for v in value["protein"]:
                    if isinstance(v, str):
                        continue
                    else:
                        Protein_list.append(v)
                        Rel_protein.append([HMDB_NO, "protein", v])

            elif key == "ontology":
                pass

            else:
                HMDB_dict[key] = value
        HMDB_dict_list.append(HMDB_dict)

    suma = 0  # 总节点数量
    triple_factor = []  # 用于存储三元组
    Rel_num = []  # 用于存储每种关系的数量

    for Rel_i in [Rel_secondary_accession,
                Rel_synonyms,
                Rel_alternative_parent,
                Rel_substituent,
                Rel_external_descriptor,
                Rel_taxonomy,
                Rel_external_property,
                Rel_predicted_property,
                Rel_spectrum,
                Rel_cellular_location,
                Rel_biospecimen_location,
                Rel_tissue,
                Rel_pathway,
                Rel_concentration,
                Rel_reference_concentration,
                Rel_reference_disease,
                Rel_disease,
                Rel_general_reference,
                Rel_protein]:
        if Rel_i:
            if isinstance(Rel_i[0], str):
                length = len(list(set(Rel_i)))
            else:
                length = len(drop_duplicate(Rel_i))
            suma += length
            length_str = str(length)
            Rel_num.append(length_str)
            ## 写入三元组
            for sn, rel, en in drop_duplicate(Rel_i):
                try:
                    triple_factor.append([sn, rel, en])
                except:
                    pass

    print(f"Total Relationship Num:{suma}")

    Node_list = []
    for node_i in HMDB_list:
        node_type = "HMDB"
        Node_list.append([node_i, node_type])

    for node_i in Substituent_list:
        node_type = "Substituent"
        Node_list.append([node_i, node_type])

    for node_i in Name_Synonyms:
        node_type = "Synonyms"
        Node_list.append([node_i, node_type])
    
    with open('data/triples.txt', 'wt', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(triple_factor)
        
    with open('data/info.txt', 'wt', newline='', encoding='utf-8') as info_file:
        info_writer = csv.writer(info_file, delimiter='\t')
        info_writer.writerows(Rel_num)
        
    with open('data/entities.txt', 'wt', newline='', encoding='utf-8') as info_file:
        info_writer = csv.writer(info_file, delimiter='\t')
        info_writer.writerows(Node_list)

    return 'data/triples.txt'

create_triples("/Users/colton/Desktop/代谢组学汇总/HMKG-Progress/HMKG/hmkg/HMDB/data/hmdb_metabolites.json")

