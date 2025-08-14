import sys
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.chat_history import BaseChatMessageHistory ##  Abstract interface for consistent history handling
from langchain_community.chat_message_histories import ChatMessageHistory ## Concrete implementation for storing conversation turns
from langchain_community.vectorstores import FAISS ## High-performance vector store for document retrieval
from langchain_core.runnables.history import RunnableWithMessageHistory ## Wrapper that adds conversation memory to any chain

from langchain.chains import create_history_aware_retriever, create_retrieval_chain ## Retriever that considers conversation history when searching documents || Combines document retrieval with answer generation
from langchain.chains.combine_documents import create_stuff_documents_chain ## Processes retrieved documents into a cohesive answer

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType

load_dotenv()

class ConversationalRAG:
    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - session_id: Unique identifier for isolating different user conversations.
        - retriever: Pre-built document search system from the src\single_document_chat\data_ingestion.py pipeline.
        - Senior thought: "How do I handle multiple users having different conversations simultaneously?" Session isolation is critical - each conversation needs its own memory and context.
        - Like a receptionist managing multiple meeting rooms - each room (session) has its own conversation that shouldn't interfere with others.'''
    def __init__(self, session_id: str, retriever):
        
        self.log = CustomLogger().get_logger(__name__)
        self.session_id = session_id
        self.retriever = retriever

        try:
            
            ''' -  Loads the language model through a dedicated helper method. 
                -  Error handling: The helper method has its own error boundaries.
                -  Reusability: Other methods might need to reload the LLM. '''
            self.llm = self._load_llm()
            
            ''' - Retrieves specialized prompts for different AI tasks from a centralized prompt registry.
                - contextualize_prompt: Helps AI understand how current questions relate to conversation history. 
                - qa_prompt: Guides AI in generating answers from retrieved documents'''
            self.contextualize_prompt = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            ''' - Creates a document retriever that considers conversation history when searching for relevant information.
                - Why history-aware: Traditional retrievers only look at the current question. 
                                     This upgraded version understands that "What about safety requirements?" might refer to a previous discussion about chemical plants.
                - User asks: "What about maintenance costs?"
                - System looks at chat history: Previous question was about "industrial pumps"
                - Retriever searches for: "What about maintenance costs for industrial pumps?"
                - Much more relevant results! '''
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, 
                self.retriever, 
                self.contextualize_prompt
            )
            
            self.log.info(
                "1. [ __init__() ] in [ src\single_document_chat\_retrieval.py ]. Created history-aware retriever", 
                session_id=session_id
            )

            ''' - Builds the core question-answering pipeline by connecting document retrieval with answer generation.
                - qa_chain: Takes documents + question → generates grounded answer.
                - rag_chain: Takes question → retrieves relevant docs → generates answer. '''
            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
            
            self.log.info(
                "1. [ __init__() ] in [ src\single_document_chat\_retrieval.py ]. Created RAG chain", 
                session_id=session_id
            )

            ''' - Wraps the RAG pipeline with conversation memory, specifying exactly how different types of messages are handled.
                - input_messages_key="input": User questions come in as "input" 
                - history_messages_key="chat_history": Previous conversation stored as "chat_history"
                - output_messages_key="answer": AI responses saved as "answer" 
                - Senior thought: "How do I make complex AI systems work together without data getting lost or mixed up?" Explicit key mappings create clear contracts between components.
                
                - rag_chain -> Wraps your existing RAG chain (the combination of retrieval + question-answering) with conversation memory capabilities
                - _get_session_history -> Provides a function that returns the appropriate chat history for each session ID 
                - input_messages_key -> Tells the wrapper that user questions come in under the key "input" in the chain's input dictionary
                - history_messages_key -> Specifies where in the chain's input the conversation history should be injected
                - output_messages_key -> Tells the wrapper where to find the AI's response in the chain's output so it can save it to conversation history '''
            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            self.log.info(
                "1. [ __init__() ] in [ src\single_document_chat\_retrieval.py ]. Wrapped chain with message history", 
                session_id=session_id
            )

        except Exception as e:
            self.log.error(
                "1. [ __init__() ] in [ src\single_document_chat\_retrieval.py ]. Error initializing ConversationalRAG", 
                error=str(e), 
                session_id=session_id
            )
            raise DocumentPortalException("1. [ __init__() ] in [ src\single_document_chat\_retrieval.py ]. Failed to initialize ConversationalRAG", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            self.log.info("2. [ _load_llm() ] in [ src\single_document_chat\_retrieval.py ]. LLM loaded successfully", class_name=llm.__class__.__name__)
            return llm
        except Exception as e:
            self.log.error("2. [ _load_llm() ] in [ src\single_document_chat\_retrieval.py ]. Error loading LLM via ModelLoader", error=str(e))
            raise DocumentPortalException("2. [ _load_llm() ] in [ src\single_document_chat\_retrieval.py ]. Failed to load LLM", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Defines a private method that returns or creates conversation history for a specific session, with explicit type hints for both input and output.
        - The return type annotation -> BaseChatMessageHistory creates a contract - callers know they'll get an object that implements the BaseChatMessageHistory interface, regardless of the concrete implementation. '''
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        try:
            ''' - Checks if the session store dictionary exists in Streamlit's session state before attempting to access it. 
                - Defensive programming - never assume shared infrastructure exists. Streamlit session state persists between page refreshes, but starts empty on first load.
                
                - Initializes an empty dictionary to hold all user conversation histories if one doesn't exist.'''
            if "store" not in st.session_state:
                st.session_state.store = {}

            ''' - Checks if this specific session already has a conversation history stored.
                - This is the "namespace isolation" pattern. Think of it like apartment buildings - each apartment (session) has its own space, but they share the same building (session_state). '''
            if session_id not in st.session_state.store:
                
                ''' - Creates a new, empty conversation history for this session and stores it in the session dictionary.'''
                st.session_state.store[session_id] = ChatMessageHistory()
                self.log.info(
                "3. [ _get_session_history() ] in [ src\single_document_chat\_retrieval.py ]. New chat session history created", 
                session_id=session_id
            )

            return st.session_state.store[session_id]
        except Exception as e:
            self.log.error("3. [ _get_session_history() ] in [ src\single_document_chat\_retrieval.py ]. Failed to access session history", session_id=session_id, error=str(e))
            raise DocumentPortalException("3. [ _get_session_history() ] in [ src\single_document_chat\_retrieval.py ]. Failed to retrieve session history", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Defines a public method that takes a file system path to a pre-built FAISS index and returns a configured document retriever. 
        - The method separates index creation from index loading—FAISS indexes are expensive to build but cheap to load. This allows offline index creation and fast runtime loading.'''
    def load_retriever_from_faiss(self, index_path: str):
        try:
            ''' - Creates an embedding model instance that can convert text into high-dimensional vectors for similarity comparison. 
                - Dependency consistency—the embedding model used to load the FAISS index MUST be identical to the one used to create it. Vector dimensions and model architectures must match exactly. '''
            embeddings = ModelLoader().load_embeddings()
            
            ''' - Validates that the provided path exists and is actually a directory before attempting to load the FAISS index. 
                - Early failure detection—better to fail fast with a clear error message than to let FAISS throw cryptic internal errors when it can't find index files. '''
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            ''' - Loads a pre-built FAISS index from disk and associates it with the embedding model for vector operations. 
                - Stateful reconstruction—FAISS indexes store vectors but not the embedding logic. You need to reunite the stored vectors with the embedding model that created them. '''
            vectorstore = FAISS.load_local(index_path, embeddings)
            self.log.info("4. [ load_retriever_from_faiss() ] in [ src\single_document_chat\_retrieval.py ]. Loaded retriever from FAISS index", index_path=index_path)
            
            ''' - Converts the FAISS vector store into a retriever with specific search behavior—similarity search returning the 5 most relevant documents.
                - Interface standardization—the retriever interface is consistent across all vector stores (FAISS, Pinecone, Chroma), making the RAG system portable.
                - Similarity search is the most common and reliable vector search method—it finds documents with similar semantic meaning regardless of exact keyword matches. 
                - Context window optimization—5 documents typically provide enough context for good answers without exceeding LLM context limits or adding too much noise.'''
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        except Exception as e:
            self.log.error("4. [ load_retriever_from_faiss() ] in [ src\single_document_chat\_retrieval.py ]. Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("4. [ load_retriever_from_faiss() ] in [ src\single_document_chat\_retrieval.py ]. Error loading retriever from FAISS", sys)
        
    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Defines the main public interface that takes a user's question as a string and returns the AI-generated answer as a string.
        - Simple contract complexity hiding—despite the intricate RAG pipeline underneath (retrieval, history management, LLM invocation), the public API is intentionally trivial. This is the hallmark of good architecture.'''
    def invoke(self, user_input: str) -> str:
        try:
            
            ''' - Executes the complete RAG pipeline by passing the user input and session configuration to the pre-built chain of operations. 
                - Wraps the user's question in a dictionary with the key "input" to match the chain's expected input format.
                - Runtime parameterization—the same chain instance can serve multiple users by injecting different session IDs at execution time. This is far more efficient than creating separate chains per user. '''
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}}
            )
            
            ''' - Safely extracts the AI-generated answer from the chain's response dictionary, providing a fallback if the "answer" key doesn't exist. 
                - Defensive data access—the .get() method prevents KeyError exceptions if the chain response format changes or if something goes wrong in the pipeline. '''
            answer = response.get("answer", "No answer.")

            ''' - Checks if the extracted answer is empty, None, or falsy, indicating that the AI didn't generate a meaningful response. '''
            if not answer:
                self.log.warning("Empty answer received", session_id=self.session_id)

            self.log.info(
                "5. [ invoke() ] in [ src\single_document_chat\_retrieval.py ]. Chain invoked successfully", 
                session_id=self.session_id, 
                user_input=user_input, 
                answer_preview=answer[:150]
            )
            
            return answer

        except Exception as e:
            self.log.error("5. [ invoke() ] in [ src\single_document_chat\_retrieval.py ]. Failed to invoke conversational RAG", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("5. [ invoke() ] in [ src\single_document_chat\_retrieval.py ]. Failed to invoke RAG chain", sys)
    
    ''' *********************************************************************************************************************************************************************************************************************** '''