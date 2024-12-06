from utils.api import DBPediaAPI
from verbalization.base import BaseVerbalization


class EnterpriseVerbalization(BaseVerbalization):
    # modify depending on the enterprise ontology
    PO_TEMPLATE = """
SELECT DISTINCT
    ?p ?o ?sLabel ?pLabel (COALESCE(?oLabel, STR(?o)) AS ?oLabel)
WHERE {{
    {entity} ?p ?o. 
    FILTER(STRSTARTS(STR(?p), STR(ns1:)))
  
    OPTIONAL {{
        {entity} rdfs:label ?sLabel.
    }}
    OPTIONAL {{
        ?p rdfs:label ?pLabel.
    }}
    OPTIONAL {{
        ?o rdfs:label ?oLabel.
    }}
}}
"""
    SP_TEMPLATE = """
SELECT DISTINCT
    ?s ?p (COALESCE(?sLabel, STR(?s)) AS ?sLabel) ?pLabel ?oLabel
WHERE {{
    {{ 
        ?s ?p {entity}.
        FILTER(STRSTARTS(STR(?p), STR(ns1:)))
    }}

    OPTIONAL {{
        ?s rdfs:label ?sLabel.
    }}
    OPTIONAL {{
        ?p rdfs:label ?pLabel.
    }}
    OPTIONAL {{
        {entity} rdfs:label ?oLabel.
    }}
}}
"""

    def __init__(
        self,
        turtle_file_path: str,
        model_name="jinaai/jina-embeddings-v3",
        model_kwargs={"trust_remote_code": True},
        query_model_encode_kwargs={},
        passage_model_encode_kwargs={},
    ) -> None:
        super().__init__(
            model_name,
            model_kwargs,
            query_model_encode_kwargs,
            passage_model_encode_kwargs,
            turtle_file_path=turtle_file_path,
        )
