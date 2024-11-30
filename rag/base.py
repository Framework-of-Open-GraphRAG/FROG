import torch, os, json
from IPython.display import HTML, display
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from xml.sax.saxutils import escape
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.output_parsers import (
    ResponseSchema,
    StructuredOutputParser,
    PydanticOutputParser,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers.prompts import NAIVE_FIX_PROMPT


from pydantic import BaseModel, Field
from typing import List

from few_shots import (
    INTENT_CLASSIFICATION_FEW_SHOTS,
)
from utils.helper import contains_multiple_entities, separate_camel_case

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseGraphRAG:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = DEVICE,
        local: str = True,
        max_new_tokens: int = 1500,
        always_use_generate_sparql: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local = local

        # To be defined in child class
        self.api = None
        self.verbalization = None
        self.property_retrieval = None

        model_kwargs = {
            "temperature": 0,
            "device": self.device,
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
        }
        if self.local:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=HF_TOKEN
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, token=HF_TOKEN
            )
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                **model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        else:
            self.tokenizer = None
            self.model = None
            llm = HuggingFaceEndpoint(
                repo_id=self.model_name,
                **model_kwargs,
                cache=False,
                huggingfacehub_api_token=HF_TOKEN,
            )
        self.chat_model = ChatHuggingFace(llm=llm)
        self.always_use_generate_sparql = always_use_generate_sparql

    def get_propertty_domain_range(self, property_uri: str) -> dict[str, str]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def handle_parsing_error(
        self,
        llm_chain: LLMChain,
        output_parser,
        messages: list[dict],
        question: str,
        chat_history_placeholder: str = "chat_history",
        try_threshold: int = 10,
    ) -> tuple[any, list[str]]:
        fix_llm_chain = NAIVE_FIX_PROMPT | self.chat_model | StrOutputParser()
        completion = llm_chain.invoke(
            {chat_history_placeholder: messages, "input": question}
        )
        messages.append(HumanMessage(content=question))

        completion_parsed = None
        while try_threshold > 0:
            try:
                completion_parsed = output_parser.parse(completion)
                break
            except Exception as e:
                display(
                    HTML(f"""<code style='color: red;'>{(escape(str(e)))}</code>""")
                )
                try_threshold -= 1

                try:
                    completion = fix_llm_chain.invoke(
                        {
                            "instructions": output_parser.get_format_instructions(),
                            "completion": completion,
                            "error": repr(e),
                        }
                    )
                except Exception as e:
                    display(
                        HTML(f"""<code style='color: red;'>{(escape(str(e)))}</code>""")
                    )
                    break

        messages.append(AIMessage(content=completion))
        if completion_parsed is not None:
            return completion_parsed, messages
        return None, messages

    def transform_to_factoid_question(
        self, question: str, try_threshold: int = 10
    ) -> str:
        response_schemas = [
            ResponseSchema(
                name="question",
                description="Factoid question transformed from user's question or instruction",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        query_reformulation_chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are transforming user query into a factoid question or instruction. A factoid question is a type of question that seeks a brief, factual answer, typically related to a specific piece of information such as a date, name, or location. The answers are usually objective and can be verified, often being short and to the point. For example:
- "Who is the president of the United States?"
- "When did World War II end?"
- "What is the capital of France?"

These questions don't require long explanations or complex reasoning and are often used in information retrieval tasks and natural language processing systems to extract concise factual information from large datasets.
Given an input query, convert it to a a factoid question. No pre-amble.

Based on the query given, transform from it and return the factoid question in the format below.
{format_instructions}""",
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    "{input}",
                ),
            ]
        )
        query_reformulation_chat_prompt = query_reformulation_chat_prompt.partial(
            format_instructions=format_instructions
        )

        llm_chain = (
            query_reformulation_chat_prompt | self.chat_model | StrOutputParser()
        )
        messages = []
        response, messages = self.handle_parsing_error(
            llm_chain, output_parser, messages, question, try_threshold=try_threshold
        )
        if response is None:
            return question
        return response["question"]

    def classify_intent_is_global(self, question: str, try_threshold: int = 10) -> bool:
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=INTENT_CLASSIFICATION_FEW_SHOTS,
        )

        class Intent(BaseModel):
            """Intent classification result."""

            is_global: bool = Field(
                ...,
                description="Whether the query is global or local.",
            )

        output_parser = PydanticOutputParser(pydantic_object=Intent)
        format_instructions = output_parser.get_format_instructions()

        query_intent_chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Your task is to classify user queries as either global or local.
- A **global** query asks for general or broad information or scope, which usually involves the usage of aggregate functions in the query like COUNT. For example, "How many films did Tom Cruise starred in?"
- A **local** query asks for specific information from a particular entity. For example, "What is the capital of France?"
Based on the query given, decide if it is global or local and return the classification in the format below.
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
        query_intent_chat_prompt = query_intent_chat_prompt.partial(
            format_instructions=format_instructions
        )

        llm_chain = query_intent_chat_prompt | self.chat_model | StrOutputParser()
        messages = []
        intent, messages = self.handle_parsing_error(
            llm_chain, output_parser, messages, question, try_threshold=try_threshold
        )
        if intent is None:
            return True
        return intent.is_global

    def extract_entity(
        self,
        question: str,
        chat_prompt_template: ChatPromptTemplate,
        try_threshold: int = 10,
    ) -> list[str]:
        class Entities(BaseModel):
            """Identifying information about entities."""

            names: List[str] = Field(
                ...,
                description="All the entities appearing in the text, sorted by importance.",
            )

        output_parser = PydanticOutputParser(pydantic_object=Entities)
        format_instructions = output_parser.get_format_instructions()

        final_prompt = chat_prompt_template.partial(
            format_instructions=format_instructions
        )
        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        entities, _ = self.handle_parsing_error(
            llm_chain, output_parser, [], question, try_threshold=try_threshold
        )
        if entities is None:
            return []
        return entities.names

    def get_most_appropriate_entity_uri(
        self,
        entity: str,
        question: str,
        retrieved_entities: list[dict],
        resource_model,
        chat_prompt_template: ChatPromptTemplate,
        try_threshold: int = 10,
    ) -> str:
        output_parser = PydanticOutputParser(pydantic_object=resource_model)
        format_instructions = output_parser.get_format_instructions()
        final_prompt = chat_prompt_template.partial(
            format_instructions=format_instructions,
            retrieved_entities=retrieved_entities,
            question=question,
        )

        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        resource, _ = self.handle_parsing_error(
            llm_chain, output_parser, [], entity, try_threshold=try_threshold
        )
        return resource

    def generate_related_properties(
        self,
        question: str,
        related_property_model,
        chat_prompt_template: ChatPromptTemplate,
        try_threshold: int = 10,
    ) -> list[str]:
        output_parser = PydanticOutputParser(pydantic_object=related_property_model)
        format_instructions = output_parser.get_format_instructions()
        final_prompt = chat_prompt_template.partial(
            format_instructions=format_instructions,
        )

        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        related_property, _ = self.handle_parsing_error(
            llm_chain,
            output_parser,
            [],
            question,
            try_threshold=try_threshold,
        )
        return related_property

    def _parse_property_context_string(self, ontology: dict[str, list[str]]) -> str:
        def parallel_search(key, name):
            try:
                domain_range = self.get_propertty_domain_range(name)
                return key, name, domain_range
            except NotImplementedError:
                return key, name, NotImplemented

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(parallel_search, key, c)
                for key, cand in ontology.items()
                if cand
                for c in cand
                if key != "classes"
            ]

            result = {}
            for future in futures:
                key, name, domain_range = future.result()
                result.setdefault(key, []).append(
                    {"name": name, "domain_range": domain_range}
                )

        properties_context = ""
        if ontology.get("classes", None):
            properties_context += f"    - classes: {ontology['classes']}\n"
        for key, value in result.items():
            properties_context += f"    - {key}: \n"
            for prop in value:
                name = prop["name"]
                domain_range = prop["domain_range"]
                if domain_range == NotImplemented:
                    properties_context += f"        - {name}\n"
                elif domain_range:
                    properties_context += f"        - {name}: {domain_range}\n"
                else:
                    properties_context += f"        - {name}: No domain and range\n"

        return properties_context

    def generate_sparql(
        self,
        question: str,
        entities: list[str],
        chat_prompt_template: ChatPromptTemplate,
        use_cot: bool = True,
        verbose: bool = False,
        try_threshold: int = 10,
    ) -> tuple[str, list[dict[str, str]]]:
        class SPARQLQueryResults(BaseModel):
            """Represents the chain of thoughts and the SPARQL query generated to answer the user's question."""

            if use_cot:
                thoughts: List[str] = Field(
                    ...,
                    description="Thoughts to generate SPARQL query to answer the user's question.",
                )
            sparql: str = Field(
                ...,
                description="SPARQL query to answer the user's question.",
            )

        output_parser = PydanticOutputParser(pydantic_object=SPARQLQueryResults)
        format_instructions = output_parser.get_format_instructions()

        resources = ""
        for entity in entities:
            resources += f"    - All possible resources URIs for {entity} are "
            resources += str(self.api.get_entities(entity, k=3)[0])
            resources += "\n"
        if verbose:
            resources_tmp = escape(resources).replace("\n", "<br/>")
            display(
                HTML(
                    f"""<code style='color: green;'>Retrieved Resources: <br/>{resources_tmp}</code>"""
                )
            )

        related_properties = self.generate_related_properties(question)[:3]
        for i in range(len(related_properties)):
            related_properties[i] = separate_camel_case(related_properties[i]).lower()
        if verbose:
            display(
                HTML(
                    f"""<code style='color: green;'>Generated Related Properties: {escape(str(related_properties))}</code>"""
                )
            )

        ontology = self.property_retrieval.get_related_candidates(
            question, property_candidates=related_properties, threshold=0.6
        )
        properties_context = self._parse_property_context_string(ontology)
        if verbose:
            properties_context_tmp = escape(properties_context).replace("\n", "<br/>")
            display(
                HTML(
                    f"""<code style='color: green;'>Retrieved Ontology: <br/>{properties_context_tmp}</code>"""
                )
            )

        final_prompt = chat_prompt_template.partial(
            resources=resources,
            ontology=properties_context,
            format_instructions=format_instructions,
        )
        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        messages = []

        curr_question = question
        while True:
            sparql_query_result, messages = self.handle_parsing_error(
                llm_chain,
                output_parser,
                messages,
                curr_question,
                try_threshold=try_threshold,
            )
            if (
                sparql_query_result is None
                or sparql_query_result.sparql == ""
                or sparql_query_result.sparql is None
            ):
                display(
                    HTML(
                        f"""<code style='color: green;'>Sorry, we are not supported with this kind of query yet.</code>"""
                    )
                )
                return None, []
            if verbose:
                if use_cot:
                    thoughts_tmp = escape(str(sparql_query_result.thoughts))
                    display(
                        HTML(f"""<code style='color: green;'>{thoughts_tmp}</code>""")
                    )
                sparql_tmp = escape(sparql_query_result.sparql).replace("\n", "<br/>")
                display(HTML(f"""<code style='color: green;'>{sparql_tmp}</code>"""))
            try:
                result, err = self.api.execute_sparql(sparql_query_result.sparql)
            except Exception as e:
                display(HTML(f"""<code style='color: red;'>{e}</code>"""))
                result, err = [], str(e)

            if len(result) == 0 and try_threshold > 0:
                # failed
                try_threshold -= 1
                if verbose:
                    display(
                        HTML(f"""<code style='color: green;'>Trying again...</code>""")
                    )

                curr_question = f"""The SPARQL query you generated above to answer '{question}' is wrong {f"and it produces this error: '{err}'" if err is not None else "because it produces empty results"}, please fix the query and generate SPARQL again! You may try to use another property or restucture the query.
DO NOT include any explanations or apologies in your responses. No pre-amble. Make sure to still answer using chain of thoughts and structure based on the format instruction defined in system prompt."""
            else:
                # success
                break
        return sparql_query_result.sparql, result

    def run(
        self,
        question: str,
        use_cot: bool = True,
        use_transform_factoid: bool = True,
        output_uri=False,
        verbose: int = 0,
        try_threshold: int = 10,
    ) -> tuple[str, str, list[dict[str, str]]]:
        if use_transform_factoid:
            factoid_question = self.transform_to_factoid_question(question)
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Factoid Question: {escape(factoid_question)}</code>"""
                    )
                )
        else:
            factoid_question = question

        extracted_entities = self.extract_entity(factoid_question)
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>Entities: {escape(str(extracted_entities))}</code>"""
                )
            )

        if self.always_use_generate_sparql:
            intent_is_global = True
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Intent is always global because always_use_generate_sparql is set to True</code>"""
                    )
                )
        else:
            intent_is_global = self.classify_intent_is_global(question)
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Intent is global: {escape(str(intent_is_global))}</code>"""
                    )
                )

        if not intent_is_global and contains_multiple_entities(question):
            intent_is_global = True
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Use SPARQL generation because the question contains multiple entities.</code>"""
                    )
                )

        if not intent_is_global:
            entity = extracted_entities[0]
            retrieved_resources = self.api.get_entities(entity, k=3)[0]
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Retrieved Resources: {escape(str(retrieved_resources))}</code>"""
                    )
                )
            entity_uri = self.get_most_appropriate_entity_uri(
                entity, factoid_question, retrieved_resources
            )
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Entity URI: {escape(entity_uri)}</code>"""
                    )
                )

            is_error = False
            try:
                result, similarities = self.verbalization.run(
                    factoid_question, entity_uri, output_uri=output_uri
                )
                if verbose == 1:
                    display(
                        HTML(
                            f"""<code style='color: green;'>Result: {escape(str(result))}<br/>Similarities: {escape(str(similarities))}</code>"""
                        )
                    )
            except Exception as e:
                is_error = True
                if verbose == 1:
                    display(
                        HTML(f"""<code style='color: red;'>{escape(str(e))}</code>""")
                    )
            if not is_error and similarities >= 0.65 and len(result) > 0:
                return factoid_question, "", result

        few_shots = deepcopy(self.generate_sparql_few_shot_messages)
        if not use_cot:
            for fs in few_shots:
                output = json.loads(fs["output"])
                output.pop("thoughts", None)
                fs["output"] = json.dumps(output, indent=4)

        query, result = self.generate_sparql(
            factoid_question,
            extracted_entities,
            few_shots=few_shots,
            use_cot=use_cot,
            verbose=verbose > 0,
            try_threshold=try_threshold,
        )
        return factoid_question, query, result

    def chat(
        self,
        question: str,
        use_cot: bool = True,
        verbose: int = 0,
        try_threshold: int = 10,
    ) -> str:
        factoid_question, _, dbpedia_context = self.run(
            question, use_cot=use_cot, verbose=verbose, try_threshold=try_threshold
        )
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>{escape(str(dbpedia_context))}</code>"""
                )
            )

        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Do not say "according to the context" or something like that, just answer directly with full sentence to the question using the context. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {input} 
Context: {dbpedia_context} 
Answer:""",
                ),
            ]
        )
        if type(dbpedia_context) == list:
            if len(dbpedia_context) > 0:
                context_str = f'The answer of "{factoid_question}" is '
                for c in dbpedia_context[:50]:
                    for k in c.keys():
                        context_str += k + " = " + c[k] + ", "
                context_str = context_str[:-2] + "."
            else:
                context_str = "I don't know"
        else:
            context_str = f'The answer of "{factoid_question}" is {dbpedia_context}'
        final_prompt = final_prompt.partial(dbpedia_context=context_str)

        llm_chain = final_prompt | self.chat_model | StrOutputParser()

        response = llm_chain.invoke({"input": factoid_question})
        return response
