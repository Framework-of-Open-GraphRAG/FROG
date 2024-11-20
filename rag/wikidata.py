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
    EXTRACT_ENTITY_FEW_SHOTS,
    WIKIDATA_GENERATE_SPARQL_FEW_SHOTS,
    GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
)
from utils.api import WikidataAPI
from verbalization import WikidataVerbalization
from property_retrieval import WikidataPropertyRetrieval
from .base import BaseGraphRAG

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WikidataGraphRAG(BaseGraphRAG):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = DEVICE,
        local: str = True,
        max_new_tokens: int = 1500,
        property_retrieval: Optional[WikidataPropertyRetrieval] = None,
        generate_sparql_few_shot_messages: Optional[List[dict]] = None,
        always_use_generate_sparql: bool = False,
    ) -> None:
        super().__init__(
            model_name, device, local, max_new_tokens, always_use_generate_sparql
        )
        self.api = WikidataAPI()
        self.verbalization = WikidataVerbalization(
            model_name="multi-qa-mpnet-base-cos-v1"
        )
        if generate_sparql_few_shot_messages is None:
            self.generate_sparql_few_shot_messages = WIKIDATA_GENERATE_SPARQL_FEW_SHOTS
        else:
            self.generate_sparql_few_shot_messages = generate_sparql_few_shot_messages
        df_properties = pd.read_csv("./data/wikidata_ontology/properties.csv")
        if property_retrieval is None:
            self.property_retrieval = WikidataPropertyRetrieval(df_properties)
        else:
            self.property_retrieval = property_retrieval

    #     def get_propertty_domain_range(self, property_uri: str) -> dict[str, str]:
    #         query = f"""SELECT ?domain ?range
    # WHERE {{
    #     {property_uri} rdfs:domain ?domain ;
    #                    rdfs:range ?range .
    # }}
    # """
    #         return self.api.execute_sparql_to_df(query).drop_duplicates().to_dict("records")

    def extract_entity(self, question, try_threshold=10):
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=EXTRACT_ENTITY_FEW_SHOTS,
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """- You are an entity extractor.
- Extract the entities from the given question! DO NOT hallucinate and only provide entities that are present in the question.
- These entities usage is to find the most appropriate entity ID from wikidata to be used in SPARQL queries.
- If there is no entity in the question, return empty list.
- Sort the entities based on the importance of the entity in the question.
- ONLY return the entities. DO NOT return anything else.
- DO NOT include adjectives like 'Highest', 'Lowest', 'Biggest', etc in the entity.
- DO NOT provide any extra information, for instance explanation inside a brackets like '(population)', '(area)', '(place)', '(artist)', etc
- DO NOT include any explanations or apologies in your responses.
- Remove all stop words, including conjunctions like 'and' and prepositions like 'in' and 'on' from the extracted entity.
- Make the entity singular, not plural. For instance, if the entity is foods, then transform it into food.
- Even if there is only one entity, alwayas return as a list.

Based on the query given, extract the entities from it and return the extracted entities in the format below.
{format_instructions}""",
                ),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        return super().extract_entity(question, chat_prompt_template, try_threshold)

    def get_most_appropriate_entity_uri(
        self,
        entity: str,
        question: str,
        retrieved_entities: list[dict],
        try_threshold: int = 10,
    ) -> str:
        class Entity(BaseModel):
            """
            Represents the most appropriate Wikidata entity ID selected from a given list of retrieved entities.
            This ID is used in Wikidata queries to accurately answer the user's question.
            """

            id: str = Field(
                ...,
                description="Wikidata Entity ID",
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the entity given, find the most appropriate entity ID from the list of retrieved entities given to be used in Wikidata queries to answer the given question! ONLY return the entity ID from the list of retrieved entities given. DO NOT return anything else and DO NOT hallucinate. DO NOT include any explanations or apologies in your responses.
Based on the entity given, get the most appropriate entity ID from it and return the ID in the format below.
{format_instructions}""",
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    """Retrieved entities:
{retrieved_entities}

Question:
{question}

Entity: 
{input}

Entity ID:""",
                ),
            ]
        )

        parsed_entity = super().get_most_appropriate_entity_uri(
            entity,
            question,
            retrieved_entities,
            Entity,
            chat_prompt_template,
            try_threshold,
        )

        if parsed_entity is None:
            return None
        return parsed_entity.id.replace("dbr:", "http://dbpedia.org/resource/")

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
            """
            Represents a list of Wikidata property labels relevant to answering a specific question posed by the user.
            These properties are selected based on their appropriateness for extracting the required information
            to provide an accurate and concise answer.
            """

            properties: List[str] = Field(
                ...,
                description="List of Wikidata property label that is appropriate to answer the given question by user.",
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate 3 Wikidata property label that is appropriate to answer the given question by user. DO NOT include the ID. DO NOT include any explanations or apologies in your responses. No pre-amble.
            
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
                    """You are a Wikidata SPARQL generator.
- Based on the context given, generate SPARQL query for Wikidata that would answer the user's question!
- You will also be provided with the entities with its IDs, property candidates. You are only able to generate SPARQL query from the given context. Please determine to use the most appropriate one.
- To generate the SPARQL, you can utilize the information from the given Entity IDs. You do not have to use it, but if it can help you to determine the ID of the entity, you can use it.
- USE the URI from resources given if you need to query more specific entity. On the other hand, USE classes from ontology if it's more general.
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
- Entities retrieved:
{resources}
- Property candidates retrieved:
{ontology}

Based on the query and context given, generate the SPARQL query from it and return the SPARQL query in the format below.
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

    def run(
        self,
        question: str,
        use_cot: bool = True,
        verbose: int = 0,
        try_threshold: int = 10,
    ):
        return super().run(
            question,
            use_cot=use_cot,
            use_transform_factoid=False,
            verbose=verbose,
            try_threshold=try_threshold,
        )
