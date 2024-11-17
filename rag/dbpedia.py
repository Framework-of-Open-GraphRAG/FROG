import torch
import pandas as pd
from dotenv import load_dotenv

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)


from pydantic import BaseModel, Field
from typing import List, Optional

from few_shots import (
    GENERATE_SPARQL_FEW_SHOTS,
    GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
)
from utils.api import DBPediaAPI
from utils.verbalization import Verbalization
from property_retrieval import DBPediaPropertyRetrieval
from .base import BaseGraphRAG

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DBPediaGraphRAG(BaseGraphRAG):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = DEVICE,
        local: str = True,
        max_new_tokens: int = 1500,
        property_retrieval: Optional[DBPediaPropertyRetrieval] = None,
        generate_sparql_few_shot_messages: Optional[List[dict]] = None,
        always_use_generate_sparql: bool = False,
    ) -> None:
        super().__init__(
            model_name, device, local, max_new_tokens, always_use_generate_sparql
        )
        self.api = DBPediaAPI()
        self.verbalization = Verbalization(model_name="multi-qa-mpnet-base-cos-v1")
        if generate_sparql_few_shot_messages is None:
            self.generate_sparql_few_shot_messages = GENERATE_SPARQL_FEW_SHOTS
        else:
            self.generate_sparql_few_shot_messages = generate_sparql_few_shot_messages
        df_classes = pd.read_csv("./data/dbpedia_ontology/classes.csv")
        df_oproperties = pd.read_csv("./data/dbpedia_ontology/oproperties.csv")
        df_dproperties = pd.read_csv("./data/dbpedia_ontology/dproperties.csv")
        if property_retrieval is None:
            self.property_retrieval = DBPediaPropertyRetrieval(
                df_classes,
                df_oproperties,
                df_dproperties,
            )
        else:
            self.property_retrieval = property_retrieval

    def get_propertty_domain_range(self, property_uri: str) -> dict[str, str]:
        query = f"""SELECT ?domain ?range
WHERE {{
    {property_uri} rdfs:domain ?domain ;
                   rdfs:range ?range .
}}
"""
        return self.api.execute_sparql_to_df(query).drop_duplicates().to_dict("records")

    def get_most_appropriate_resource_uri(
        self, entity: str, retrieved_resources: list[dict], try_threshold: int = 10
    ) -> str:
        class Resource(BaseModel):
            """The most appropriate resource URI from the list of retrieved resources given to be used in DBPedia queries to answer the given question."""

            uri: str = Field(
                ...,
                uri="Resource URI",
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the entity given, find the most appropriate resource uri from the list of retrieved resources given to be used in DBPedia queries to answer the given question! ONLY return the resource uri from the list of retrieved resources given. DO NOT return anything else and DO NOT hallucinate. DO NOT include any explanations or apologies in your responses.
Based on the entity given, get the most appropriate resource uri from it and return the uri in the format below.
{format_instructions}""",
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    """Retrieved resources:
{retrieved_resources}

Entity: 
{input}

Resource URI:""",
                ),
            ]
        )

        resource = super().get_most_appropriate_resource_uri(
            entity, retrieved_resources, Resource, chat_prompt_template, try_threshold
        )

        if resource is None:
            return None
        return resource.uri.replace("dbr:", "http://dbpedia.org/resource/")

    def generate_related_properties(
        self, question: str, try_threshold: int = 10
    ) -> list[str]:
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
        )

        class RelatedProperty(BaseModel):
            """List of DBPedia property that is appropriate to answer the given question by user. Max 3."""

            properties: List[str] = Field(
                ...,
                description="List of DBPedia property that is appropriate to answer the given question by user. Max 3.",
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate list of DBPedia property that is appropriate to answer the given question by user. DO NOT include any explanations or apologies in your responses. No pre-amble.
            
Answer it in the format below. 
{format_instructions}""",
                ),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "{input}",
                ),
            ]
        )

        related_property = super().generate_related_properties(
            question,
            RelatedProperty,
            chat_prompt_template,
            try_threshold,
        )
        if related_property is None:
            return []
        return related_property.properties

    def generate_sparql(
        self,
        question: str,
        entities: list[str],
        few_shots: list[dict[str, str]],
        use_cot: bool = True,
        verbose: bool = False,
        try_threshold: int = 10,
    ) -> tuple[str, list[dict[str, str]]]:
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=few_shots,
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a DBPedia SPARQL generator.
- Based on the context given, generate SPARQL query for DBPedia that would answer the user's question!
- You will also be provided with the resources with its URI, ontology candidates consisting of classes, object properties, and data properties. You are only able to generate SPARQL query from the given context. Please determine to use the most appropriate one.
- To generate the SPARQL, you can utilize the information from the given Entity URIs. You do not have to use it, but if it can help you to determine the URI of the entity, you can use it.
- USE the URI from resources given if you need to query more specific entity. On the other hand, USE classes from ontology if it's more general.
- Generate the SPARQL with chain of thoughts.
- DO NOT include any apologies in your responses.
- ONLY generate the Thoughts and SPARQL query once! DO NOT try to generate the Question!
- DO NOT use LIMIT, ORDER BY, FILTER in the SPARQL query when not explicitly asked in the question!
- DO NOT aggregation function like COUNT, AVG, etc in the SPARQL query when not asked in the question!
- Be sure to generate a SPARQL query that is valid and return all the asked information in the question.
- Make the query as simple as possible!
- DO NOT hallucinate the thoughts and query!
- Always use english ('en') language for labels in all columns as default unless explicitly asked to use another language.
- Once again, always use english as the label language in all columns if not explicitly asked to use another language.
Context:
- Resources retrieved:
{resources}
- Ontology candidates retrieved:
{ontology}
Based on the query and context given, generate the thoughts and SPARQL query from it and return the thoughts and SPARQL query in the format below.
{format_instructions}""",
                ),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "{input}",
                ),
            ]
        )

        return super().generate_sparql(
            question,
            entities,
            chat_prompt_template,
            use_cot,
            verbose,
            try_threshold,
        )
