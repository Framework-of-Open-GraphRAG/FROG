import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.api import BaseAPI
from utils.helper import replace_using_dict, separate_camel_case


class BaseVerbalization:
    SENTENCE_TEMPLATE = "{s}'s {p} is {o}"
    MANUAL_MAPPING_DICT = {"_": " "}
    PO_TEMPLATE = None
    SP_TEMPLATE = None

    def __init__(
        self,
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"trust_remote_code": True},
        query_model_encode_kwargs={},
        passage_model_encode_kwargs={},
    ) -> None:
        self.api = None  # Overwrite in child class
        self.model_name = model_name
        self.query_model_encode_kwargs = query_model_encode_kwargs
        self.passage_model_encode_kwargs = passage_model_encode_kwargs
        self.model = SentenceTransformer(model_name, **model_kwargs)

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

    def run(
        self, question: str, entity: str, output_uri=False
    ) -> tuple[list[dict[str, str]], float]:
        list_of_candidates, po, sp = self.get_list_of_candidates(entity)
        cands = list(list_of_candidates.values())
        question_embed = self.model.encode(question, **self.query_model_encode_kwargs)
        passages_embed = self.model.encode(cands, **self.passage_model_encode_kwargs)

        similarities = (
            self.model.similarity(question_embed, passages_embed).numpy().flatten()
        )
        similar_index = np.argmax(similarities)
        similar_score = max(similarities)

        property_used = list(list_of_candidates.keys())[similar_index]
        result = []
        for _, (p, o, _, pLabel, oLabel) in po[po["p"] == property_used].iterrows():
            label_p = pLabel if pLabel else separate_camel_case(p.split("/")[-1])
            if o.startswith("http"):
                label_o = (
                    oLabel
                    if oLabel
                    else replace_using_dict(o.split("/")[-1], self.MANUAL_MAPPING_DICT)
                )
            else:
                label_o = o
            result.append({label_p: o if output_uri else label_o})
        for _, (s, p, sLabel, pLabel, _) in sp[sp["p"] == property_used].iterrows():
            label_p = pLabel if pLabel else separate_camel_case(p.split("/")[-1])
            label_s = (
                sLabel
                if sLabel
                else replace_using_dict(s.split("/")[-1], self.MANUAL_MAPPING_DICT)
            )
            result.append({label_p: s if output_uri else label_s})
        return result, similar_score
