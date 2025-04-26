import torch, os, json, gc, re
from IPython.display import HTML, display
from dotenv import load_dotenv
from xml.sax.saxutils import escape
from copy import deepcopy
from rdflib import Graph
import googletrans
from googletrans import Translator

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
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
from utils.helper import (
    contains_multiple_entities,
    fix_query_spacing,
    separate_camel_case,
)
import time

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseGraphRAG:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = DEVICE,
        use_local_model: str = True,
        max_new_tokens: int = 1500,
        always_use_generate_sparql: bool = False,
        print_output: bool = False,
        additional_model_kwargs: dict = {},
        turtle_file_path: str = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_local_model = use_local_model
        self.print_output = print_output

        # To be defined in child class
        self.api = None
        self.verbalization = None
        self.property_retrieval = None

        self.translator = Translator()
        if turtle_file_path:
            self.graph = Graph().parse(turtle_file_path)
        else:
            self.graph = None

        model_kwargs = {
            "do_sample": False,
            "device": self.device,
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            **additional_model_kwargs,
        }
        if self.use_local_model:
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
                if self.print_output:
                    print(f"Error: {e}")
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
                    if self.print_output:
                        print(f"Error: {e}")
                    break

        messages.append(AIMessage(content=completion))
        torch.cuda.empty_cache()
        gc.collect()
        if completion_parsed is not None:
            return completion_parsed, messages
        return None, messages

    def translate(self, text: str, dest_lang="en") -> str:
        return self.translator.translate(text, dest=dest_lang).text

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
        properties_context = ""
        for key, value in ontology.items():
            if key == "classes":
                properties_context += f"    - classes: {value}\n"
            else:
                properties_context += f"    - {key}: \n"
                for prop in value:
                    properties_context += f"        - {prop}\n"
        return properties_context

    def _extract_query(self, text: str) -> str:
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

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

        start_time = time.time()
        resources = ""
        if self.api:
            for entity in entities:
                resources += f"    - All possible resources URIs for {entity} are "
                resources += str(self.api.get_entities(entity, k=5)[0])
                resources += "\n"
            if verbose:
                resources_tmp = escape(resources).replace("\n", "<br/>")
                display(
                    HTML(
                        f"""<code style='color: green;'>Retrieved Resources: <br/>{resources_tmp}</code>"""
                    )
                )
            if self.print_output:
                print("Retrieved Resources: ", resources)

        try:
            related_properties = self.generate_related_properties(question)[:3]
            for i in range(len(related_properties)):
                related_properties[i] = separate_camel_case(
                    related_properties[i]
                ).lower()
            if verbose:
                display(
                    HTML(
                        f"""<code style='color: green;'>Generated Related Properties: {escape(str(related_properties))}</code>"""
                    )
                )
            if self.print_output:
                print("Generated Related Properties: ", related_properties)
        except NotImplementedError:
            related_properties = entities

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
        if self.print_output:
            print("Retrieved Ontology: ", properties_context)
        end_time = time.time()
        print(
            f"Entity and Property retrieval for SPARQL Generation time: {end_time - start_time:.2f} seconds"
        )

        final_prompt = chat_prompt_template.partial(
            resources=resources,
            ontology=properties_context,
            format_instructions=format_instructions,
        )
        llm_chain = final_prompt | self.chat_model | StrOutputParser()
        messages = []

        start_time = time.time()
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
                if self.print_output:
                    print("Sorry, we are not supported with this kind of query yet.")
                return None, []
            if verbose:
                if use_cot:
                    thoughts_tmp = escape(str(sparql_query_result.thoughts))
                    display(
                        HTML(f"""<code style='color: green;'>{thoughts_tmp}</code>""")
                    )
                sparql_tmp = escape(sparql_query_result.sparql).replace("\n", "<br/>")
                # sparql_tmp = escape(response).replace("\n", "<br/>")
                display(HTML(f"""<code style='color: green;'>{sparql_tmp}</code>"""))
            if self.print_output:
                if use_cot:
                    print("Thoughts: ", sparql_query_result.thoughts)
                print("Generated SPARQL: ", sparql_query_result.sparql)
                # print("Generated Response: ", response)
            try:
                if self.api:
                    result, err = self.api.execute_sparql(sparql_query_result.sparql)
                else:
                    result = []
                    for res in self.graph.query(
                        fix_query_spacing(sparql_query_result.sparql)
                    ):
                        res_dct = [{k: str(v)} for k, v in res.asdict().items()]
                        result.extend(res_dct)
                    err = None
            except Exception as e:
                display(HTML(f"""<code style='color: red;'>{e}</code>"""))
                if self.print_output:
                    print("Error: ", e)
                result, err = [], str(e)

            if (
                len(result) == 0
                or (len(result) == 1 and list(result[0].values())[0] == "0")
            ) and try_threshold > 0:
                # failed
                try_threshold -= 1
                if verbose:
                    display(
                        HTML(f"""<code style='color: green;'>Trying again...</code>""")
                    )
                if self.print_output:
                    print("Trying again...")

                curr_question = f"""The SPARQL query you generated above to answer '{question}' is wrong {f"and it produces this error: '{err}'" if err is not None else "because it produces empty results"}, please fix the query and generate SPARQL again! You may try to use another property or restucture the query.
DO NOT include any explanations or apologies in your responses. No pre-amble. Make sure to still answer using chain of thoughts and structure based on the format instruction defined in system prompt."""
            else:
                # success
                break
        end_time = time.time()
        print(f"SPARQL Generation time: {end_time - start_time:.2f} seconds")
        return sparql_query_result.sparql, result

    def run(
        self,
        question: str,
        use_cot: bool = True,
        use_transform_factoid: bool = False,
        output_uri=False,
        verbose: int = 0,
        try_threshold: int = 10,
    ) -> tuple[str, str, list[dict[str, str]]]:
        if self.translator.detect(question).lang != "en":
            start_time = time.time()
            question = self.translate(question, dest_lang="en")
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Translated Question: {escape(question)}</code>"""
                    )
                )
            if self.print_output:
                print("Translated Question: ", question)
            end_time = time.time()
            print(f"Translation time: {end_time - start_time:.2f} seconds")

        if use_transform_factoid:
            factoid_question = self.transform_to_factoid_question(question)
            if verbose == 1:
                display(
                    HTML(
                        f"""<code style='color: green;'>Factoid Question: {escape(factoid_question)}</code>"""
                    )
                )
            if self.print_output:
                print("Factoid Question: ", factoid_question)
        else:
            factoid_question = question

        start_time = time.time()
        extracted_entities = self.extract_entity(factoid_question)
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>Entities: {escape(str(extracted_entities))}</code>"""
                )
            )
        if self.print_output:
            print("Entities: ", extracted_entities)

        intent_is_global = self.always_use_generate_sparql

        # if not intent_is_global and contains_multiple_entities(question):
        #     intent_is_global = True
        #     if verbose == 1:
        #         display(
        #             HTML(
        #                 f"""<code style='color: green;'>Use SPARQL generation because the question contains multiple entities.</code>"""
        #             )
        #         )
        #     if self.print_output:
        #         print(
        #             "Use SPARQL generation because the question contains multiple entities."
        #         )

        # if not intent_is_global and len(extracted_entities) > 0:
        entity = extracted_entities[0]
        if self.api:
            retrieved_resources = self.api.get_entities(entity, k=5)[0]
        else:
            retrieved_resources = self.property_retrieval.search_entities(entity, k=5)[
                ["short", "label", "score"]
            ].rename({"short": "uri"}, axis=1)
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>Retrieved Resources: {escape(str(retrieved_resources))}</code>"""
                )
            )
        if self.print_output:
            print("Retrieved Resources: ", retrieved_resources)
        entity_uri = self.get_most_appropriate_entity_uri(
            entity, factoid_question, retrieved_resources
        )
        if verbose == 1:
            display(
                HTML(
                    f"""<code style='color: green;'>Entity URI: {escape(entity_uri)}</code>"""
                )
            )
        if self.print_output:
            print("Entity URI: ", entity_uri)

        end_time = time.time()
        print(f"Entiti linking time: {end_time - start_time:.2f} seconds")

        start_time = time.time()
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
            if self.print_output:
                print("Result: ", result, "Similarities: ", similarities)
        except Exception as e:
            is_error = True
            if verbose == 1:
                display(HTML(f"""<code style='color: red;'>{escape(str(e))}</code>"""))
            if self.print_output:
                print("Error: ", e)
        end_time = time.time()
        print(f"Verbalization time: {end_time - start_time:.2f} seconds")

        if not is_error and similarities >= 0.6 and len(result) > 0:
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

    def process_context(
        self, question: str, context: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def chat(
        self,
        question: str,
        use_cot: bool = True,
        verbose: int = 0,
        try_threshold: int = 10,
    ) -> str:
        total_start_time = time.time()
        start_time = time.time()
        factoid_question, _, context = self.run(
            question, use_cot=use_cot, verbose=verbose, try_threshold=try_threshold
        )
        end_time = time.time()
        print(f"Run time: {end_time - start_time:.2f} seconds")

        start_time = time.time()
        lang_detected = self.translator.detect(question).lang
        final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Do not say "according to the context" or something like that, just answer directly with full sentence to the question using the context. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.{f" Answer in {googletrans.LANGUAGES[lang_detected]} language." if lang_detected != 'en' else ''}
Question: {{input}} 
Context: {{context}} 
Answer:""",
                ),
            ]
        )
        context_str, context = self.process_context(question, context)
        if verbose == 1:
            display(
                HTML(f"""<code style='color: green;'>{escape(str(context))}</code>""")
            )
        if self.print_output:
            print(context)

        final_prompt = final_prompt.partial(context=context_str)

        llm_chain = final_prompt | self.chat_model | StrOutputParser()

        response = llm_chain.invoke({"input": factoid_question})
        end_time = time.time()
        print(f"Answer Generation time: {end_time - start_time:.2f} seconds")
        total_end_time = time.time()
        print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
        return response
