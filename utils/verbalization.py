import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.api import DBPediaAPI
from utils.helper import replace_using_dict, separate_camel_case


class Verbalization:
    SENTENCE_TEMPLATE = "{s}'s {p} is {o}"
    MANUAL_MAPPING_DICT = {"_": " "}
    PO_TEMPLATE = """
select ?p ?o ?sLabel ?pLabel ?oLabel where {{
  <{entity}> ?p ?o .
  FILTER (
    strstarts(str(?p), "http://dbpedia.org/ontology/")
    && !contains(str(?p), "wiki")
    && (strstarts(str(?o), "http://dbpedia.org/") || isLiteral(?o))
    && ?p != <http://dbpedia.org/ontology/abstract>
  )
  FILTER (
    !isLiteral(?o) || (isLiteral(?o) && (lang(?o) = "en" || lang(?o) = ""))
  )
  
  OPTIONAL {{
    <{entity}> rdfs:label ?sLabel .
    FILTER(lang(?sLabel) = "en")
  }}
  
  OPTIONAL {{
    ?p rdfs:label ?pLabel .
    FILTER(lang(?pLabel) = "en")
  }}
  
  OPTIONAL {{
    ?o rdfs:label ?oLabel .
    FILTER(lang(?oLabel) = "en")
  }}
}}
"""
    SP_TEMPLATE = """
select ?s ?p ?sLabel ?pLabel ?oLabel where {{
  ?s ?p <{entity}> .
  FILTER (
    strstarts(str(?p), "http://dbpedia.org/ontology/")
    && !contains(str(?p), "wiki")
  )
  
  OPTIONAL {{
    ?s rdfs:label ?sLabel .
    FILTER(lang(?sLabel) = "en")
  }}
  
  OPTIONAL {{
    ?p rdfs:label ?pLabel .
    FILTER(lang(?pLabel) = "en")
  }}
  
  OPTIONAL {{
    <{entity}> rdfs:label ?oLabel .
    FILTER(lang(?oLabel) = "en")
  }}
}}
"""

    def __init__(self, model_name="multi-qa-mpnet-base-cos-v1") -> None:
        self.api = DBPediaAPI()
        self.model = SentenceTransformer(model_name)

    def get_po(self, entity: str) -> pd.DataFrame:
        query = self.PO_TEMPLATE.format(entity=entity)
        df = self.api.execute_sparql_to_df(query).drop_duplicates()
        if df.empty:
            return pd.DataFrame(columns=["p", "o", "sLabel", "pLabel", "oLabel"])
        return df

    def get_sp(self, entity: str) -> pd.DataFrame:
        query = self.SP_TEMPLATE.format(entity=entity)
        df = self.api.execute_sparql_to_df(query).drop_duplicates()
        if df.empty:
            return pd.DataFrame(columns=["s", "p", "sLabel", "pLabel", "oLabel"])
        return df

    def get_list_of_candidates(self, entity: str):
        po, sp = self.get_po(entity), self.get_sp(entity)

        candidates = dict()

        # no duplicate properties
        curr_p = None
        for _, (p, o, sLabel, pLabel, oLabel) in po.iterrows():
            label_s = (
                sLabel
                if sLabel
                else replace_using_dict(entity.split("/")[-1], self.MANUAL_MAPPING_DICT)
            )
            label_p = pLabel if pLabel else separate_camel_case(p.split("/")[-1])

            if label_p != curr_p:
                curr_p = label_p
                if o.startswith("http"):
                    label_o = (
                        oLabel
                        if oLabel
                        else replace_using_dict(
                            o.split("/")[-1], self.MANUAL_MAPPING_DICT
                        )
                    )
                else:
                    label_o = o
                candidates[p] = self.SENTENCE_TEMPLATE.format(
                    s=label_s, p=label_p, o=label_o
                )

        curr_p = None
        for _, (s, p, sLabel, pLabel, oLabel) in sp.iterrows():
            label_s = (
                sLabel
                if sLabel
                else replace_using_dict(s.split("/")[-1], self.MANUAL_MAPPING_DICT)
            )
            label_p = pLabel if pLabel else separate_camel_case(p.split("/")[-1])
            label_o = (
                oLabel
                if oLabel
                else replace_using_dict(entity.split("/")[-1], self.MANUAL_MAPPING_DICT)
            )

            if label_p != curr_p:
                curr_p = label_p
                candidates[p] = self.SENTENCE_TEMPLATE.format(
                    s=label_s, p=label_p, o=label_o
                )

        return candidates, po, sp

    def run(self, question: str, entity: str) -> tuple[list[dict[str, str]], float]:
        question_embed = self.model.encode(question)

        list_of_candidates, po, sp = self.get_list_of_candidates(entity)
        cands = list(list_of_candidates.values())
        passages_embed = self.model.encode(cands)

        similarities = (
            self.model.similarity(question_embed, passages_embed).numpy().flatten()
        )
        similar_index = np.argmax(similarities)
        similar_score = max(similarities)

        property_used = list(list_of_candidates.keys())[similar_index]
        result = []
        for _, (p, o, _, pLabel, _) in po[po["p"] == property_used].iterrows():
            label_p = pLabel if pLabel else separate_camel_case(p.split("/")[-1])
            result.append({label_p: o})
        for _, (s, p, _, pLabel, _) in sp[sp["p"] == property_used].iterrows():
            label_p = pLabel if pLabel else separate_camel_case(p.split("/")[-1])
            result.append({label_p: s})
        return result, similar_score
