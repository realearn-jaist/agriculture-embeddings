
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


model = torch.load("model/model2.reasonEModel.1000", map_location="cpu")

with open("model/model2.entityMap", "r") as f:
    emap = f.readlines()[1:-1]
emap = [l.split("\t") for l in emap]
emap = {k:int(v.strip()) for k, v in emap}

with open("model/model2.unaryConceptMap", "r") as f:
    urmap = f.readlines()[1:-1]
urmap = [l.split("\t") for l in urmap]
urmap = {k:int(v.strip()) for k, v in urmap}

with open("model/model2.binaryConceptMap", "r") as f:
    brmap = f.readlines()[1:-1]
brmap = [l.split("\t") for l in brmap]
brmap = {k:int(v.strip()) for k, v in brmap}

# eq_idx = brmap["http://www.w3.org/2002/07/owl#equivalentClass"]
# subclassof_idx = brmap["http://www.w3.org/2000/01/rdf-schema#subClassOf"]


with torch.no_grad():
    # entity_idx = torch.LongTensor([pato320_idx, green_idx, pale_green_idx, ricedo127_idx, co_320_200_idx, ricedo4_idx, ricedo126_idx, ricedo15_idx, hash1_idx, ricedo32_idx])
    # concept_idx = torch.LongTensor([restric_idx])
    # relation_idx = torch.LongTensor([eq_idx, subclassof_idx, some_idx])

    relation100_idx = torch.arange(0, len(brmap), dtype=torch.long)
    # pato320_vec, green_vec, pale_green_vec, ricedo127_vec, co_320_200_vec, ricedo4_vec, ricedo126_vec, ricedo15_vec, hash1_vec, ricedo32_vec = model.entityEmbed(entity_idx)
    # restric_vec, = model.uConceptEmbed(concept_idx)

    relation10h_vec = model.bConceptHEmbed(relation100_idx)
    relation10t_vec = model.bConceptTEmbed(relation100_idx)
    # eqh_vec, subclassofh_vec, someh_vec = model.bConceptHEmbed(relation_idx)
    # eqt_vec, subclassoft_vec, somet_vec = model.bConceptTEmbed(relation_idx)

    all_relation_idx = torch.arange(0, len(brmap), dtype=torch.long)
    all_entity_idx = torch.arange(0, len(emap), dtype=torch.long)
    all_entity_vec = model.entityEmbed(all_entity_idx)

    relation10h_vec = model.bConceptHEmbed(all_relation_idx)
    relation10t_vec = model.bConceptTEmbed(all_relation_idx)

    plt.figure()
    plt.title("Histrogram of values in all entity embeddings")
    plt.hist(torch.log10(all_entity_vec.abs().flatten() + torch.finfo(torch.float64).eps), bins=15)
    plt.xlabel("log10(values in entity embeddings)")
    plt.savefig("qe_plot/hist.png")

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

    plt.close("all")
