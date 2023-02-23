from bioservices import kegg,chebi
from pubchempy import Compound


def get_KEGG_cpd_info(cid,keys_list=["REACTION","PATHWAY","MODULE","ENZYME"]):
    k=kegg.KEGG()
    entities=set()
    triples=set()
    parse_result=k.parse(k.get(cid))
    keys_list=list(set(keys_list)&set(parse_result.keys()))
    cid="kegg_id:"+str(cid)
    for key in keys_list:
        if key=="REACTION":
            for _ in parse_result["REACTION"]:
                _="reaction_id:"+str(_)
                entities.add((_,"Reaction"))
                triples.add((cid,"has_reaction",_))
        if key=="ENZYME":
            for _ in parse_result["ENZYME"]:
                _="enzyme_id:"+str(_)
                entities.add((_,"Enzyme"))
                triples.add((cid,"has_enzyme",_))
        if key=="PATHWAY":
            for _ in list(parse_result["PATHWAY"].keys()):
                _="pathway_id:"+str(_)
                entities.add((_,"Pathway"))
                triples.add((cid,"has_pathway",_))
        if key=="MODULE":
            for _ in list(parse_result["MODULE"].keys()):
                _="module_id:"+str(_)
                entities.add((_,"Module"))
                triples.add((cid,"has_module",_))
    
    return entities,triples


def get_CHEBI_cpd_info(cid,keys_list=["OntologyParents"]):
    entities=set()
    triples=set()
    c=chebi.ChEBI()
    parse_result=c.getCompleteEntity(cid)
    cid="chebi_id:"+str(cid)
    for key in keys_list:
        if key=="OntologyParents":
            _=parse_result["OntologyParents"]
            for i in _:
                entities.add((str(i["chebiId"]),"Chebi_id"))
                triples.add((cid,str(i["type"]),str(i["chebiId"])))
    return entities,triples


def get_PubChem_cpd_info(cid,keys_list=["h_bond_acceptor_count","h_bond_donor_count","heavy_atom_count"]):
    entities=set()
    triples=set()
    parse_result=Compound.from_cid(cid).to_dict()
    keys_list=list(set(keys_list)&set(parse_result.keys()))
    cid="pubchem_id:"+str(cid)
    for key in keys_list:
        if key=="h_bond_acceptor_count":
            _="atom_count:"+str(parse_result["h_bond_acceptor_count"])
            entities.add((_,"h_bond_acceptor_count"))
            triples.add((cid,"h_bond_acceptor_count",_))
        if key=="h_bond_donor_count":
            _="atom_count:"+str(parse_result["h_bond_donor_count"])
            entities.add((_,"h_bond_donor_count"))
            triples.add((cid,"h_bond_donor_count",_))
        if key=="heavy_atom_count":
            _="atom_count:"+str(parse_result["heavy_atom_count"])
            entities.add((_,"heavy_atom_count"))
            triples.add((cid,"heavy_atom_count",_))
    return entities,triples
    