import owlready2 as owl
import rdflib
import pandas as pd
from typing import cast
from os import PathLike
from pathlib import Path
from pykeen.triples import TriplesFactory, CoreTriplesFactory

def save_dataset_to_tsv(splited_dataset: CoreTriplesFactory, dataset: TriplesFactory, save_path: PathLike):
    l = []
    for h, r, t in splited_dataset.mapped_triples:
        h = cast(int, h.item())
        r = cast(int, r.item())
        t = cast(int, t.item())
        l.append((dataset.entity_id_to_label[h], 
                  dataset.relation_id_to_label[r], 
                  dataset.entity_id_to_label[t]))
        
    df = pd.DataFrame(l)
    df.to_csv(save_path, header=False, index=False, sep="\t")

onto = owl.get_ontology("RiceDO-Version2.owl")
onto.load()
onto.save("all.rdf")

g = rdflib.Graph()
g.parse("all.rdf", format="xml")

data = [(s, p, o) for s, p, o in g]
df = pd.DataFrame(data)
df.to_csv("all.tsv", sep='\t', index=False, header=False)

dataset = TriplesFactory.from_path("all.tsv")
training, testing, validation = dataset.split([0.8, 0.1, 0.1], random_state=247803864)

save_dataset_to_tsv(training, dataset, Path("train.tsv"))
save_dataset_to_tsv(validation, dataset, Path("validate.tsv"))
save_dataset_to_tsv(testing, dataset, Path("test.tsv"))
