import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

''' *********************************************************************************************************************************************************************************************************************** '''
''' - ConversationalRAG orchestrates a 3-stage LCEL graph:
        1. Contextualize the user’s question using chat history.
            - Room 1 (Contextualizer): Clean up and rephrase the problem.
        2. Retrieve relevant docs from FAISS using embeddings.
            - Room 2 (Retriever): Fetch the right tools/materials.
        3. Answer the question using retrieved context + input + history.
            - Room 3 (Answerer): Use the materials to craft a clear solution. '''
class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """
    ''' *********************************************************************************************************************************************************************************************************************** '''
    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id

            ''' - Load the LLM client via a helper method.
                - self.contextualize_prompt:
                    - Retrieve the prompt template used to rewrite user questions using chat_history (contextualization).
                    - Users ask context-dependent questions in chats (“What about its specs?”). Retrieval works better on a fully specified, stand-alone question.
                    - Separation of concerns—one prompt is dedicated to rewriting, not answering.
                - self.qa_prompt:
                    - Retrieve the prompt template for answering questions using retrieved context plus the original input.
                    - The answer stage requires a different instruction set than the rewrite stage. '''
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            ''' - Store the retriever if provided; otherwise keep it None. 
                - Supports lazy initialization—don’t require or load FAISS at construction time.
                - Delay heavy dependencies until needed (“pay-as-you-go”). Makes the object usable in environments where the index isn’t ready yet.'''
            self.retriever = retriever
            
            ''' - Placeholder for the LCEL graph (pipeline).
                - The chain depends on the retriever; without it, the graph would be incomplete.
                - Model the lifecycle explicitly—unbuilt state is represented as None. Build when dependencies exist.'''
            self.chain = None
            
            ''' - If a retriever was injected, build the LCEL graph immediately so the instance is invokable. '''
            if self.retriever is not None:
                self._build_lcel_chain()

            self.log.info("1. [ __init__() ] from [ src\\document_chat\\retrieval.py ]. ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            self.log.error("1. [ __init__() ] from [ src\\document_chat\\retrieval.py ]. Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("1. [ __init__() ] from [ src\\document_chat\\retrieval.py ]. Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------
    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Load FAISS vectorstore from disk and build retriever + LCEL chain. 
        - Public API to attach retrieval to the RAG pipeline by loading a FAISS index from disk and building an LCEL chain. 
        - Separates heavy retrieval setup from constructor; supports lazy loading, multi-tenant indexes, and DI for tests. '''
    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        try:
            
            ''' - Validate the index directory exists before attempting load.'''
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"2. [ load_retriever_from_faiss() ] from [ src\\document_chat\\retrieval.py ]. FAISS index directory not found: {index_path}")

            embeddings = ModelLoader().load_embeddings()
            
            ''' - Load a FAISS vector store from disk with the given embeddings.'''
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # ok if you trust the index
            )

            ''' - Provide default search parameters if none supplied. '''
            if search_kwargs is None:
                search_kwargs = {"k": k}

            ''' - Wrap the vector store as a retriever with configured search behavior.
                - Standardizes the interface for LCEL graphs, which expect a retriever-like component. '''
            self.retriever = vectorstore.as_retriever(
                search_type=search_type, 
                search_kwargs=search_kwargs
            )
            
            ''' - Rebuild the LCEL pipeline now that the retriever exists. '''
            self._build_lcel_chain()

            self.log.info(
                "2. [ load_retriever_from_faiss() ] from [ src\\document_chat\\retrieval.py ]. FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                k=k,
                session_id=self.session_id,
            )
            return self.retriever

        except Exception as e:
            self.log.error("2. [ load_retriever_from_faiss() ] from [ src\\document_chat\\retrieval.py ]. Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("2. [ load_retriever_from_faiss() ] from [ src\\document_chat\\retrieval.py ]. Loading error in ConversationalRAG", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Invoke the LCEL pipeline.
        - Public entry point to run the end-to-end RAG pipeline for a single query.
        - Separates the two user-controlled inputs: the current turn (user_input) and the prior turns (chat_history). Optional chat history keeps the API ergonomic for one-off calls.'''
    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        try:
            
            ''' - Guard clause to ensure the pipeline is built. 
                - Failing fast with a clear message beats obscure NoneType errors deep in LCEL execution. '''
            if self.chain is None:
                raise DocumentPortalException(
                    "3. [ invoke() ] from [ src\\document_chat\\retrieval.py ]. RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            
            ''' - Normalize None to empty list. '''
            chat_history = chat_history or []
            
            ''' - Prepare the exact keys the LCEL graph expects.
                - Your _build_lcel_chain wired itemgetter("input") and itemgetter("chat_history"), so the graph expects a dict with those keys.'''
            payload = {"input": user_input, "chat_history": chat_history}
            
            ''' - Execute the LCEL graph: rewrite question → retrieve docs → answer with LLM. '''
            answer = self.chain.invoke(payload)
            
            ''' - Handle the empty/None response case gracefully. 
                - LLMs or chains can return empty strings/None in edge cases. Returning a friendly string makes the UI stable and debuggable.'''
            if not answer:
                self.log.warning(
                    "3. [ invoke() ] from [ src\\document_chat\\retrieval.py ]. No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            
            self.log.info(
                "3. [ invoke() ] from [ src\\document_chat\\retrieval.py ]. Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        
        except Exception as e:
            self.log.error("3. [ invoke() ] from [ src\\document_chat\\retrieval.py ]. Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("3. [ invoke() ] from [ src\\document_chat\\retrieval.py ]. Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------
    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Private helper that initializes the Large Language Model client and returns it. '''
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            self.log.info("4. [ _load_llm() ] from [ src\\document_chat\\retrieval.py ]. LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            self.log.error("4. [ _load_llm() ] from [ src\\document_chat\\retrieval.py ]. Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Produces a single string by concatenating multiple document objects, separating each with two newline characters.
        - Iterates through docs, extracts a string from each element (prefer d.page_content), and joins the strings with a blank line between items.
        - LLM-friendly context: Most LLM prompts expect a plain text “context” block. 
          This function converts a list of retrieved documents into a single text chunk that the answer-generation prompt can consume directly.
        - @staticmethod:
            - Declares that this method does not depend on instance state (self).'''
    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' - Think “three rooms on a conveyor belt”: 
            - Room 1: Contextualizer rewrites the user’s question using chat history.
            - Room 2: Retriever fetches relevant docs for that rewritten question.
            - Room 3: Answerer uses retrieved context + original input + chat history to produce the final answer.
        - The | operator is LCEL’s “pipe,” passing outputs to the next step. '''
    def _build_lcel_chain(self):
        try:
            
            ''' - What it does: Precondition check — can’t build a chain without a retriever.
                - Why: Fails fast instead of letting a later step blow up with a cryptic error.'''
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            ''' - What it does:
                    - Shapes the incoming payload to just the keys needed (“input”, “chat_history”).
                    - Renders the contextualization prompt.
                    - Calls the LLM to rewrite the question.
                    - Parses the output to a plain string.'''
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            ''' - Feeds the rewritten string into the retriever to get top-k docs.
                - Converts the list of Document objects into a single string with _format_docs.
                - Answer generation wants a single “context” text block, not a list.
                - Keeping formatting in a tiny function makes the pipeline clean and replaceable. '''
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            ''' - Composes the final stage: provide the retrieved context, the original user input, and the chat history to the QA prompt.
                - Calls the LLM to generate the answer and parses it to a string.'''
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            self.log.info("6. [ _build_lcel_chain() ] from [ src\\document_chat\\retrieval.py ]. LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            self.log.error("6. [ _build_lcel_chain() ] from [ src\\document_chat\\retrieval.py ]. Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
    
    ''' *********************************************************************************************************************************************************************************************************************** '''