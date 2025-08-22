import sys
import os
from typing import List,Optional

from dotenv import load_dotenv
from operator import itemgetter


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

class ConversationalRAG:

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Creates a new RAG instance tied to one chat session. Accepts an external retriever.
        - session_id isolates logs and async workloads per user.'''
    def __init__(self, session_id: str, retriever = None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            self.llm = self._load_llm()
            
            ''' - Pulls two distinct prompts from a central registry.
                - RAG needs two prompts:
                    1. Contextualize the question based on retrieved documents. A question rewriter that embeds chat history
                    2. Answer the question using the context. A QA prompt that combines context and user input.
                - Both prompts are ChatPromptTemplates, which can handle chat history and user input.'''
            self.contextualize_prompt:ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt:ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            
            ''' - A missing retriever makes every downstream call useless; better to crash immediately.'''
            if retriever is None:
                raise ValueError("Retriever Cannot Be None")
            self.retriever = retriever
            
            self._build_lcel_chain()
            self.log.info(
                "1. [__init__() ] in [ src\multi_document_chat\retrieval.py ]. ConversationalRAG Initialized", 
                session_id = self.session_id
            )

        except Exception as e:
            self.log.error("1. [__init__() ] in [ src\multi_document_chat\retrieval.py ]. Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("1. [__init__() ] in [ src\multi_document_chat\retrieval.py ]. Initialization error in ConversationalRAG", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Public utility that turns an on-disk FAISS index into an in-memory retriever object.
        - Lets us hot-swap or version knowledge bases without rebuilding the whole RAG pipeline. '''
    def load_retriever_from_faiss(self,index_path:str):

        try:
            embeddings = ModelLoader().load_embeddings()
            
            ''' - Checks if the FAISS index directory exists.
                - If not, raises an error to prevent silent failures.'''
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS Index Directory not Found: {index_path}")
            
            ''' - Reads raw index files sitting at index_path.
                - Re-creates the in-memory FAISS data structure (all vectors, metadata, search trees).
                - Binds the exact embeddings function so any new query is projected into the same vector space the index was built in.
                - allow_dangerous_deserialization=True disables safety checks around Python pickle loading.
                - Unlocking your own house door with your key = safe. Using the same unlocked door in a public hotel = reckless.'''
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            ''' - Converts raw FAISS into a high-level LangChain retriever that returns the top-5 most similar documents.
                - search_type="similarity" means it uses cosine similarity to find the closest vectors.
                - search_kwargs={"k": 5} limits results to the top-5 matches.'''
            self.retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":5})
            
            self.log.info(
                "2. [load_retriever_from_faiss() ] in [ src\multi_document_chat\retrieval.py ]. FAISS Retriver Loaded Successfylly", 
                index_path = index_path, 
                session_id = self.session_id
            )

            return self.retriever

        except Exception as e:
            self.log.error("2. [load_retriever_from_faiss() ] in [ src\multi_document_chat\retrieval.py ]. Failed to Load Retriever from FAISS", error=str(e))
            raise DocumentPortalException("2. [load_retriever_from_faiss() ] in [ src\multi_document_chat\retrieval.py ]. Initialization error in Load Retriever from FAISS", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' - Declares a callable entry point that accepts the latest user question plus any prior conversation.'''
    def invoke(self, user_input:str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        try:
            
            ''' - If no chat history is provided, initializes it as an empty list.
                - This allows the method to handle both new questions and follow-ups seamlessly.'''
            chat_history = chat_history or []
            
            ''' - Bundles the two pieces of context into a dictionary.
                - The chain expects a dictionary with "input" and "chat_history" keys.
                - The LangChain pipeline (self.chain) consumes dictionaries keyed exactly this way.'''
            payload = {"input" : user_input, "chat_history": chat_history}
            
            ''' - Invokes the LCEL chain with the user input and chat history.
                - The chain processes the input, retrieves relevant documents, and generates an answer.
                    1. Rewrites the question in context of chat history.
                    2. Retrieves top-k documents via the retriever.
                    3. Feeds everything to the QA prompt + LLM.
                    4. Parses the raw LLM output into a plain string.'''
            answer = self.chain.invoke(payload)
            
            ''' - Handles the edge case where the LLM returns nothing.
                - If no answer is generated, logs a warning and returns a default message.
                - This prevents downstream errors in the chat interface.
                - Always check for empty responses to avoid breaking the chat flow.'''
            if not answer:
                self.log.warning(
                    "No Answer Generated", 
                    user_input = user_input, 
                    session_id = self.session_id
                )
                return "no answer generated."
            
            self.log.info(
                "3. [invoke() ] in [ src\multi_document_chat\retrieval.py ]. Chain Invoked Successfully",
                session_id = self.session_id,
                user_input=user_input,
                answer_preview = answer[:150]
            )
            
            return answer
        
        except Exception as e:
            self.log.error("3. [invoke() ] in [ src\multi_document_chat\retrieval.py ]. Failed to Invoke", error=str(e))
            raise DocumentPortalException("3. [invoke() ] in [ src\multi_document_chat\retrieval.py ]. Initialization error in Invoking", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            
            ''' - Immediately checks the loader didn’t return None.'''
            if not llm:
                raise ValueError("LLM Could not be loaded")

            self.log.info(
                "4. [_load_llm() ] in [ src\\multi_document_chat\\retrieval.py ]. "
                "LLM Loaded Successfully",
                session_id=self.session_id
            )
            return llm

        except Exception as e:
            self.log.error(
                "4. [_load_llm() ] in [ src\\multi_document_chat\\retrieval.py ]. "
                "Failed to Load LLM",
                error=str(e)
            )
            raise DocumentPortalException(
                "4. [_load_llm() ] in [ src\\multi_document_chat\\retrieval.py ]. "
                "Initialization error in Loading LLM",
                sys
            )


    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' - Receives a list of document objects (docs)—each produced by the retriever.
        - Extracts the raw text of every document via d.page_content.
        - Concatenates those texts into one string, inserting two newline characters between each chunk.
        - @staticmethod -> The routine needs no instance state, so mark it static for clarity and testability. '''
    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' - This private helper constructs one LangChain Expression Language (LCEL) pipeline that glues three big tasks together:
            1. Contextualize the question based on retrieved documents.
            2. Answer the question using the context.
            3. Parse the final answer into a string.
        - The chain is built once during initialization and reused for every user query.'''
    def _build_lcel_chain(self):
        try:
            
            ''' - itemgetter() -> Pulls input & chat_history from the caller-supplied payload.
                - self.contextualize_prompt -> Prompt that asks the LLM to rewrite the new question using prior messages.
                - self.llm -> The LLM that generates the rewritten question.
                - StrOutputParser() -> Converts the LLM’s output into a plain string.'''
            question_rewriter = (
                { 
                    "input" : itemgetter("input"),
                    "chat_history" : itemgetter("chat_history")
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            
            ''' - self.retriever -> The retriever that  maps the (possibly rewritten) query to top-k document chunks.
                - self._format_docs -> Static method that stitches those chunks together with double newlines so the LLM sees clear boundaries.'''
            retrieve_docs = self.retriever | self._format_docs
            
            ''' - itemgetter() -> Pulls the rewritten question and chat history from the payload.
                - self.qa_prompt -> The prompt that instructs the LLM to answer the question using the context.
                - self.llm -> The LLM that generates the final answer.
                - StrOutputParser() -> Converts the LLM’s output into a plain string.'''
            self.chain = (
                {
                    "context" : retrieve_docs,
                    "input" : itemgetter("input"),
                    "chat_history" : itemgetter("chat_history")
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            self.log.info(
                "6. [_build_lcel_chain() ] in [ src\multi_document_chat\retrieval.py ]. LCEL Chain Build Successful", 
                session_id = self.session_id
            )
            
        except Exception as e:
            self.log.error("6. [_build_lcel_chain() ] in [ src\multi_document_chat\retrieval.py ]. Failed to build LCEL chain", error=str(e))
            raise DocumentPortalException("6. [_build_lcel_chain() ] in [ src\multi_document_chat\retrieval.py ]. Initialization error in building LCEL chain", sys)
    
    ''' *********************************************************************************************************************************************************************************************************************** '''
