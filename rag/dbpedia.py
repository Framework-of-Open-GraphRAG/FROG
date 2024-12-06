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
    DBPEDIA_GENERATE_SPARQL_FEW_SHOTS,
    WIKIDATA_DBPEDIA_EXTRACT_ENTITY_FEW_SHOTS,
    GENERATE_RELATED_PROPERTIES_FEW_SHOTS,
)
from utils.api import DBPediaAPI
from verbalization import DBPediaVerbalization
from property_retrieval import DBPediaPropertyRetrieval
from .base import BaseGraphRAG

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DBPediaGraphRAG(BaseGraphRAG):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = DEVICE,
        use_local_model: str = True,
        max_new_tokens: int = 1500,
        property_retrieval: Optional[DBPediaPropertyRetrieval] = None,
        generate_sparql_few_shot_messages: Optional[List[dict]] = None,
        always_use_generate_sparql: bool = False,
        use_local_weaviate_client: bool = True,
        print_output: bool = False,
        additional_model_kwargs: dict = {},
    ) -> None:
        super().__init__(
            model_name,
            device,
            use_local_model,
            max_new_tokens,
            always_use_generate_sparql,
            print_output,
            additional_model_kwargs,
        )
        self.api = DBPediaAPI()
        self.verbalization = DBPediaVerbalization(
            model_name="jinaai/jina-embeddings-v3",
            query_model_encode_kwargs={
                "task": "retrieval.query",
                "prompt_name": "retrieval.query",
            },
            passage_model_encode_kwargs={
                "task": "retrieval.passage",
                "prompt_name": "retrieval.passage",
            },
        )
        if generate_sparql_few_shot_messages is None:
            self.generate_sparql_few_shot_messages = DBPEDIA_GENERATE_SPARQL_FEW_SHOTS
        else:
            self.generate_sparql_few_shot_messages = generate_sparql_few_shot_messages
        if property_retrieval is None:
            df_classes = pd.read_csv("./data/dbpedia_ontology/classes.csv")
            df_oproperties = pd.read_csv("./data/dbpedia_ontology/oproperties.csv")
            df_dproperties = pd.read_csv("./data/dbpedia_ontology/dproperties.csv")
            self.property_retrieval = DBPediaPropertyRetrieval(
                df_classes,
                df_oproperties,
                df_dproperties,
                embedding_model_name="jinaai/jina-embeddings-v3",
                is_local_client=use_local_weaviate_client,
            )
        else:
            self.property_retrieval = property_retrieval

    def extract_entity(self, question, try_threshold=10):
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=WIKIDATA_DBPEDIA_EXTRACT_ENTITY_FEW_SHOTS,
        )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """- You are an entity extractor.
- Extract the entities from the given question! DO NOT hallucinate and only provide entities that are present in the question.
- These entities usage is to find the most appropriate resource URI from dbpedia to be used in SPARQL queries.
- If there is no entity in the question, return empty list.
- Sort the entities based on the importance of the entity in the question.
- ONLY return the entities. DO NOT return anything else.
- DO NOT include adjectives like 'Highest', 'Lowest', 'Biggest', etc in the entity.
- DO NOT provide any extra information, for instance explanation inside a brackets like '(population)', '(area)', '(place)', '(artist)', etc
- DO NOT include any explanations or apologies in your responses.
- Remove all stop words, including conjunctions like 'and' and prepositions like 'in' and 'on' from the extracted entity.
- Make the entity singular, not plural. For instance, if the entity is foods, then transform it into food.
- DO NOT separate Proper Names, e.g. 'Amazon River' should be returned as 'Amazon River'.
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
        class Resource(BaseModel):
            """
            Represents the most appropriate DBPedia resource URI selected from a given list of retrieved resources.
            This URI is used in DBPedia queries to accurately answer the user's question.
            """

            uri: str = Field(
                ...,
                description="DBPedia Resource URI",
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
{retrieved_entities}

Question:
{question}

Entity: 
{input}

Resource URI:""",
                ),
            ]
        )

        resource = super().get_most_appropriate_entity_uri(
            entity,
            question,
            retrieved_entities,
            Resource,
            chat_prompt_template,
            try_threshold,
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
            """
            Represents a list of DBPedia properties relevant to answering a specific question posed by the user.
            These properties are selected based on their appropriateness for extracting the required information
            to provide an accurate and concise answer.
            """

            properties: List[str] = Field(
                ...,
                description="List of DBPedia property that is appropriate to answer the given question by user.",
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Generate 3 DBPedia property that is appropriate to answer the given question by user. DO NOT include any explanations or apologies in your responses. No pre-amble.
            
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
                    """# INSTRUCTIONS
You are an assistant trained to generate DBPedia SPARQL queries. Use the provided context to generate a valid SPARQL query. Based on the given entities and properties context, please generate a valid SPARQL query to answer the question! Use the URI from resources given if you need to query more specific entity. On the other hand, USE classes from ontology if it's more general. Return only the uri or literal only. Always generate following the format instruction.

# CONTEXT
- Entity: {resources}
- Ontology candidates: 
{ontology}

# FORMAT INSTRUCTIONS
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

    def process_context(
        self, question: str, context: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        if type(context) == list:
            if len(context) > 0:
                if list(context[0].values())[0].startswith("http://dbpedia.org/"):
                    context_entities = []
                    for c in context[:50]:
                        context_entities.append(
                            "<http://dbpedia.org/resource/"
                            + list(c.values())[0].split("/")[-1]
                            + ">"
                        )
                    get_label_query = f"""SELECT ?{list(c.keys())[0]} WHERE {{
  VALUES ?item {{ {" ".join(context_entities)} }}
  ?item rdfs:label ?{list(c.keys())[0]}.
  FILTER (lang(?{list(c.keys())[0]}) = "en")
}}"""
                    context, _ = self.api.execute_sparql(get_label_query)
                context_str = f'The answer of "{question}" is '
                for c in context[:50]:
                    for k, v in c.items():
                        context_str += k + " = " + v + ", "
                context_str = context_str[:-2] + "."
            else:
                context_str = "I don't know"
        else:
            context_str = f'The answer of "{question}" is {context}'
        return context_str, context

    def run(
        self,
        question: str,
        use_cot: bool = True,
        use_transform_factoid: bool = False,
        output_uri: bool = False,
        verbose: int = 0,
        try_threshold: int = 10,
    ):
        return super().run(
            question,
            use_cot=use_cot,
            output_uri=output_uri,
            use_transform_factoid=use_transform_factoid,
            verbose=verbose,
            try_threshold=try_threshold,
        )
