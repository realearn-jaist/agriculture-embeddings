import torch
import matplotlib.pyplot as plt
import seaborn as sns


def relation_plot(title, ent_vec_nameh, ent_vec_namet, rel_vec_nameh, rel_vec_namet):
    ent_vec_h, ent_name_h = ent_vec_nameh
    ent_vec_t, ent_name_t = ent_vec_namet
    rel_vec_h, rel_name_h = rel_vec_nameh
    rel_vec_t, rel_name_t = rel_vec_namet
    plt.subplot(4, 1, 1)
    plt.title(title)
    # sns.heatmap(ent_vec_h.unsqueeze(0), vmin=-1, vmax=1)
    sns.heatmap(torch.log10(ent_vec_h.abs().unsqueeze(0)), vmin=-20, vmax=0)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(ent_name_h)
    plt.subplot(4, 1, 2)
    sns.heatmap(rel_vec_h.unsqueeze(0), vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(rel_name_h)
    plt.subplot(4, 1, 3)
    # sns.heatmap(ent_vec_t.unsqueeze(0), vmin=-1, vmax=1)
    sns.heatmap(torch.log10(ent_vec_t.abs().unsqueeze(0)), vmin=-20, vmax=0)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(ent_name_t)
    plt.subplot(4, 1, 4)
    sns.heatmap(rel_vec_t.unsqueeze(0), vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(rel_name_t)

def concept_plot(title, ent_vec_name, con_vec_name):
    ent_vec, ent_name = ent_vec_name
    con_vec, con_name = con_vec_name
    plt.subplot(2, 1, 1)
    plt.title(title)
    sns.heatmap(torch.log10(ent_vec.abs().unsqueeze(0)), vmin=-20, vmax=0)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(ent_name)
    plt.subplot(2, 1, 2)
    sns.heatmap(con_vec.unsqueeze(0), vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(con_name)


model = torch.load("model/model3.reasonEModel.1500", map_location="cpu")

with open("model/model3.entityMap", "r") as f:
    emap = f.readlines()[1:-1]
emap = [l.split("\t") for l in emap]
emap = {k:int(v.strip()) for k, v in emap}

with open("model/model3.unaryConceptMap", "r") as f:
    urmap = f.readlines()[1:-1]
urmap = [l.split("\t") for l in urmap]
urmap = {k:int(v.strip()) for k, v in urmap}

with open("model/model3.binaryConceptMap", "r") as f:
    brmap = f.readlines()[1:-1]
brmap = [l.split("\t") for l in brmap]
brmap = {k:int(v.strip()) for k, v in brmap}

eq_idx = brmap["http://www.w3.org/2002/07/owl#equivalentClass"]
subclassof_idx = brmap["http://www.w3.org/2000/01/rdf-schema#subClassOf"]

# http://purl.org/ricedo#RiceDO_000067	http://www.w3.org/2002/07/owl#equivalentClass	http://purl.obolibrary.org/obo/PATO_0000320
# both of entity are equivalent class, hence their vector must be the same
pato320_idx = emap["http://purl.obolibrary.org/obo/PATO_0000320"]

# http://purl.org/ricedo#RiceDO_000070	http://www.w3.org/2000/01/rdf-schema#subClassOf	http://purl.org/ricedo#RiceDO_000067
# pale green is subclass of green
green_idx = emap["http://purl.org/ricedo#RiceDO_000067"]
pale_green_idx = emap["http://purl.org/ricedo#RiceDO_000070"] 

# http://purl.org/ricedo#RiceDO_000127	http://www.w3.org/2002/07/owl#equivalentClass	http://www.cropontology.org/rdf/CO_320:0000200
ricedo127_idx = emap["http://purl.org/ricedo#RiceDO_000127"]
co_320_200_idx = emap["http://www.cropontology.org/rdf/CO_320:0000200"] 

# http://purl.org/ricedo#RiceDO_000015	http://www.w3.org/2000/01/rdf-schema#subClassOf	http://purl.org/ricedo#RiceDO_000004
# http://purl.org/ricedo#RiceDO_000126	http://www.w3.org/2000/01/rdf-schema#subClassOf	http://purl.org/ricedo#RiceDO_000015
ricedo4_idx = emap["http://purl.org/ricedo#RiceDO_000004"]
ricedo15_idx = emap["http://purl.org/ricedo#RiceDO_000015"]
ricedo126_idx = emap["http://purl.org/ricedo#RiceDO_000126"] 

# Nd9879f5c272c4650a50452ea2bd974bf	http://www.w3.org/1999/02/22-rdf-syntax-ns#type	http://www.w3.org/2002/07/owl#Restriction
hash1_idx = emap["Nd9879f5c272c4650a50452ea2bd974bf"]
restric_idx = urmap["http://www.w3.org/2002/07/owl#Restriction"]

# Nd9879f5c272c4650a50452ea2bd974bf	http://www.w3.org/2002/07/owl#someValuesFrom	http://purl.org/ricedo#RiceDO_000032
ricedo32_idx = emap["http://purl.org/ricedo#RiceDO_000032"]
some_idx = brmap["http://www.w3.org/2002/07/owl#someValuesFrom"]

# http://purl.obolibrary.org/obo/PDO_0000003	http://www.w3.org/2000/01/rdf-schema#subClassOf	http://purl.org/ricedo#RiceDO_000006
# http://purl.obolibrary.org/obo/PDO_0000019	http://www.w3.org/2000/01/rdf-schema#subClassOf	http://purl.obolibrary.org/obo/PDO_0000003
pdo3_idx = emap["http://purl.obolibrary.org/obo/PDO_0000003"]
pdo19_idx = emap["http://purl.obolibrary.org/obo/PDO_0000019"]
ricedo6_idx = emap["http://purl.org/ricedo#RiceDO_000006"]

with torch.no_grad():
    entity_idx = torch.LongTensor([pato320_idx, green_idx, pale_green_idx, ricedo127_idx, co_320_200_idx, ricedo4_idx, ricedo126_idx, ricedo15_idx, hash1_idx, ricedo32_idx, pdo3_idx, pdo19_idx, ricedo6_idx])
    concept_idx = torch.LongTensor([restric_idx])
    relation_idx = torch.LongTensor([eq_idx, subclassof_idx, some_idx])

    all_relation_idx = torch.arange(0, len(brmap), dtype=torch.long)
    all_entity_idx = torch.arange(0, len(emap), dtype=torch.long)
    pato320_vec, green_vec, pale_green_vec, ricedo127_vec, co_320_200_vec, ricedo4_vec, ricedo126_vec, ricedo15_vec, hash1_vec, ricedo32_vec, pdo3_vec, pdo19_vec, ricedo6_vec = model.entityEmbed(entity_idx)
    all_entity_vec = model.entityEmbed(all_entity_idx)
    restric_vec, = model.uConceptEmbed(concept_idx)

    relation10h_vec = model.bConceptHEmbed(all_relation_idx)
    relation10t_vec = model.bConceptTEmbed(all_relation_idx)
    eqh_vec, subclassofh_vec, someh_vec = model.bConceptHEmbed(relation_idx)
    eqt_vec, subclassoft_vec, somet_vec = model.bConceptTEmbed(relation_idx)

    print(torch.finfo(torch.float64).eps)
    plt.figure()
    plt.title("Histrogram of values in all entity embeddings")
    plt.hist(torch.log10(all_entity_vec.abs().flatten() + torch.finfo(torch.float64).eps), bins=15)
    plt.xlabel("log10(values in entity embeddings)")
    plt.savefig("qe_plot/hist.png")

    plt.figure()
    relation_plot("Pale green issubclassof Green", 
                  (pale_green_vec, "Pale green"), 
                  (green_vec, "Green"),
                  (subclassofh_vec, "SubClassOf H"),
                  (subclassoft_vec, "SubClassOf T"))
    plt.savefig("qe_plot/pale_green_subclassof_green.png")

    plt.figure()
    relation_plot("Green equivalentClass pato320", 
                  (green_vec, "Green"),
                  (pato320_vec, "pato320"), 
                  (eqh_vec, "equivalentClass H"),
                  (eqt_vec, "equivalentClass T"))
    plt.savefig("qe_plot/green_eqclass_pato320.png")

    plt.figure()
    relation_plot("RiceDO127 equivalentClass CO_320_200",
                  (ricedo127_vec, "RiceDO127"),
                  (co_320_200_vec, "CO_320_200"),
                  (eqh_vec, "equivalentClass H"),
                  (eqt_vec, "equivalentClass T"))
    plt.savefig("qe_plot/ricedo127_eqclass_co_320_200.png")

    plt.figure()
    relation_plot("RiceDO126 subclassof RiceDO4",
                  (ricedo126_vec, "RiceDO126"),
                  (ricedo4_vec, "RiceDO4"),
                  (subclassofh_vec, "SubClassOf H"),
                  (subclassoft_vec, "SubClassOf T"))
    plt.savefig("qe_plot/ricedo126_subclassof2_ricedo4.png")
    
    plt.figure()
    relation_plot("RiceDO15 subclassof RiceDO4",
                  (ricedo15_vec, "RiceDO15"),
                  (ricedo4_vec, "RiceDO4"),
                  (subclassofh_vec, "SubClassOf H"),
                  (subclassoft_vec, "SubClassOf T"))
    plt.savefig("qe_plot/ricedo15_subclassof_ricedo4.png")
    
    plt.figure()
    relation_plot("RiceDO126 subclassof RiceDO15",
                  (ricedo126_vec, "RiceDO126"),
                  (ricedo15_vec, "RiceDO15"),
                  (subclassofh_vec, "SubClassOf H"),
                  (subclassoft_vec, "SubClassOf T"))
    plt.savefig("qe_plot/ricedo126_subclassof_ricedo15.png")

    plt.figure()
    concept_plot("974bf is restriction", 
                 (hash1_vec, "974bf"),
                 (restric_vec, "restriction"))
    plt.savefig("qe_plot/974bf_is_restriction.png")

    plt.figure()
    relation_plot("974bf someValuesFrom RiceDO32",
                  (hash1_vec, "974bf"),
                  (ricedo32_vec, "RiceDO32"),
                  (someh_vec, "someValuesFrom H"),
                  (somet_vec, "someValuesFrom T"))
    plt.savefig("qe_plot/974bf_somevaluefrom_ricedo32.png")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(relation10h_vec, vmax=1, vmin=0)
    plt.title("Head Relation Embedding")
    plt.ylabel("Each relation h vector")
    plt.xlabel("Each value in relation h vector")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(relation10t_vec, vmax=1, vmin=0)
    plt.title("Tail Relation Embedding")
    plt.ylabel("Each relation t vector")
    plt.xlabel("Each value in relation t vector")
    plt.savefig("qe_plot/rel25.png")

    # http://purl.obolibrary.org/obo/PDO_0000003	http://www.w3.org/2000/01/rdf-schema#subClassOf	http://purl.org/ricedo#RiceDO_000006
    plt.figure()
    relation_plot("PDO3 subClassOf RiceDO6",
                  (pdo3_vec, "PDO3"),
                  (ricedo6_vec, "RiceDO6"),
                  (subclassofh_vec, "subClassOf H"),
                  (subclassoft_vec, "subClassOf T"))
    plt.savefig("qe_plot/pdo3_subclassof_ricedo6.png")

    # http://purl.obolibrary.org/obo/PDO_0000019	http://www.w3.org/2000/01/rdf-schema#subClassOf	http://purl.obolibrary.org/obo/PDO_0000003
    plt.figure()
    relation_plot("PDO19 subClassOf PDO3",
                  (pdo19_vec, "PDO19"),
                  (pdo3_vec, "PDO3"),
                  (subclassofh_vec, "subClassOf H"),
                  (subclassoft_vec, "subClassOf T"))
    plt.savefig("qe_plot/pdo19_subclassof_pdo3.png")
    plt.close("all")

