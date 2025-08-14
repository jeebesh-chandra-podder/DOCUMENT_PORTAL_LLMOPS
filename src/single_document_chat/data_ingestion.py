import uuid
from pathlib import Path
import sys
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader

'''
Raw Documents → PDF files uploaded by users
Text Extraction → Content pulled from PDFs
Chunking → Text split into searchable pieces
Embedding → Chunks converted to mathematical vectors
Vector Store → Optimized storage for similarity search
Retriever → The interface that makes it all usable
'''
class SingleDocIngestor:
    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' -  Defines the constructor that runs automatically when creating a new SingleDocIngestor instance. '''
    def __init__(self,data_dir: str = "data/single_document_chat", faiss_dir: str = "faiss_index"):
        try:
            
            ''' - Creates a logger instance specifically for this class, using the module name as the logger identifier. 
                - __name__: Automatically uses the current module name (data_ingestion) as the logger name.
                - Instance variable (self.log): Makes the logger available to all methods in the class. 
                - CustomLogger(): Uses a centralized logging configuration rather than basic Python logging'''
            self.log = CustomLogger().get_logger(__name__)
            
            ''' - Converts string paths to Path objects and creates the directories if they don't exist.
                - Intuitive operations: You can use / to join paths: self.data_dir / filename
                - Built-in methods: .mkdir(), .exists(), .is_file() are cleaner than os.path equivalents.
                - parents=True: Creates parent directories if needed.
                - exist_ok=True: Doesn't fail if directory already exists. '''
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            
            ''' - Creates an instance of the ModelLoader class to handle embedding model operations.
                - Single Responsibility: The document ingestor doesn't need to know about model details.
                - Reusability: Other classes can use the same ModelLoader. '''
            self.model_loader = ModelLoader()
            
            self.log.info(
                "1. [__init__() ] in [ src\single_document_chat\data_ingestion.py ]. SingleDocIngestor initialized", 
                temp_path=str(self.data_dir), 
                faiss_path=str(self.faiss_dir)
            )
        
        except Exception as e:
            self.log.error("1. [__init__() ] in [ src\single_document_chat\data_ingestion.py ]. Failed to initialize SingleDocIngestor", error=str(e))
            raise DocumentPortalException("1. [__init__() ] in [ src\single_document_chat\data_ingestion.py ]. Initialization error in SingleDocIngestor", sys)
        
    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' - Defines a method that accepts a collection of uploaded files and processes them into a searchable format. 
        - Plural parameter name: uploaded_files suggests it can handle multiple files in one call'''
    def ingest_files(self,uploaded_files):
        try:
            
            ''' - Starts error handling and initializes an empty list to collect all processed documents.
                - Accumulator pattern: documents = [] will collect results from all files.
                - Senior thought: "If processing 10 files and file #7 fails, what should happen?" This design fails the entire batch rather than partially succeeding, which prevents inconsistent state.
                - Why not handle errors per file? Because partial success in document ingestion creates confusing user experiences - better to process all or none. 
                - This is the "All-or-Nothing Transaction" pattern - commonly used in database operations. '''
            documents = []
            
            ''' - Iterates through each file and generates a globally unique filename using timestamp + UUID.
                - session_ prefix: Makes it easy to identify these files in the file system.
                - datetime.now(timezone.utc): Uses UTC to avoid timezone confusion in distributed systems. 
                - strftime('%Y%m%d_%H%M%S'): Creates sortable timestamp (YYYYMMDD_HHMMSS).
                - uuid.uuid4().hex[:8]: Adds 8 characters of randomness to guarantee uniqueness.
                - .pdf extension: Preserves file type information'''
            for uploaded_file in uploaded_files:
                unique_filename = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                
                ''' - Creates the full file path and writes the uploaded file to disk in binary mode.
                    - with open(..., "wb"): Context manager with binary write mode.
                    - uploaded_file.read(): Reads the entire file content into memory. 
                    - The "wb" is crucial! PDFs are binary files - opening in text mode would corrupt the data. 
                    - The with statement ensures the file is properly closed even if an exception occurs during writing.'''
                temp_path = self.data_dir / unique_filename
                with open(temp_path, "wb") as f_out:
                    f_out.write(uploaded_file.read()) 
                
                self.log.info(
                    "2. [ingest_files() ] in [ src\single_document_chat\data_ingestion.py ]. PDF saved for ingestion", 
                    filename=uploaded_file.name
                )
                
                ''' - Uses LangChain's PyPDFLoader to extract text content from the PDF and adds it to the documents collection. 
                    - str(temp_path): PyPDFLoader expects string paths, not Path objects.
                    - .load(): Returns a list of Document objects (one per page typically). 
                    - .extend(): Adds all pages to the main documents list (not .append() which would add the list as a single item). '''
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)
            
            self.log.info("2. [ingest_files() ] in [ src\single_document_chat\data_ingestion.py ]. PDF files loaded", count=len(documents))
            return self._create_retriever(documents)
                
        except Exception as e:
            self.log.error("2. [ingest_files() ] in [ src\single_document_chat\data_ingestion.py ]. Document ingestion failed", error=str(e))
            raise DocumentPortalException("2. [ingest_files() ] in [ src\single_document_chat\data_ingestion.py ]. Error during file ingestion", sys)
        
    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' - Defines a private method (indicated by the underscore prefix) that transforms raw documents into a searchable retriever object.
        - Single Responsibility: This method has one job - convert documents to retrievers.
        - Internal API: Other classes shouldn't call this directly; it's part of the internal processing pipeline.
        - Encapsulation: Hides the complexity of vector store creation from external users. 
        - Senior devs think: "What operations are implementation details vs. public interface?" Document chunking, embedding, and FAISS indexing are internal concerns that users don't need to know about. '''
    def _create_retriever(self,documents):
        try:
            
            ''' - Breaks down large documents into smaller, overlapping chunks optimized for vector search and LLM processing. 
                - chunk_size=1000: Balances context preservation with embedding model limits (most models have token limits).
                - chunk_overlap=300: Ensures important information spanning chunk boundaries isn't lost.
                - RecursiveCharacterTextSplitter: Tries to split intelligently (sentences, paragraphs) before resorting to character limits. 
                - Senior thought process: "How do I maintain semantic meaning while fitting technical constraints?" The overlap creates redundancy that preserves context across boundaries.'''
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=300
            )
            chunks = splitter.split_documents(documents)
            
            self.log.info(
                "3. [_create_retriever() ] in [ src\single_document_chat\data_ingestion.py ]. Documents split into chunks", 
                count=len(chunks)
            )
            
            ''' - Retrieves the embedding model through the centralized ModelLoader.
                - Separation of Concerns: This method focuses on retriever creation; ModelLoader handles model management.
                - Senior thought: "Loading embedding models is expensive - how do I avoid doing it repeatedly?" Delegation to a specialized loader that handles caching and configuration.'''
            embeddings = self.model_loader.load_embeddings()
            
            ''' - Creates a FAISS vector store by embedding all document chunks and building a searchable index.
                - Each chunk gets converted to a high-dimensional vector (embedding).
                - FAISS builds an optimized index structure for fast similarity search.
                - Original text chunks are stored alongside their vector representations. 
                - Senior devs think: "What's the most expensive operation here?" Creating embeddings for every chunk. 
                                     "How do I make searches fast?" Use specialized data structures like FAISS instead of brute-force comparison. '''
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            
            ''' - Saves the vector store to disk and logs the operation.
                - Why persist to disk: Creating embeddings is computationally expensive -> don't repeat it.
                - Crash recovery: If the process crashes, the index survives.
                - Why log with path: Operations teams need to know where indexes are stored for monitoring disk usage, backups, and troubleshooting.
                - The str(self.faiss_dir) conversion is needed because FAISS expects string paths, not Path objects.'''
            vectorstore.save_local(str(self.faiss_dir))
            self.log.info(
                "3. [_create_retriever() ] in [ src\single_document_chat\data_ingestion.py ]. FAISS index created and saved", 
                faiss_path=str(self.faiss_dir)
            )
            
            ''' 
            Imagine you have a massive library with millions of books. A traditional librarian would help you find books by title or author. 
            A retriever is like a genius librarian who understands the meaning of your questions and can instantly find the most relevant passages from any book, 
            even if your question doesn't use the exact words in the text.

            Senior devs think: "How do I bridge the gap between human questions and document content?" 
            A retriever solves this by understanding semantic similarity, not just keyword matching.
            '''
            
            ''' - Configures the vector store as a retriever with specific search parameters.
                - search_type="similarity": Uses cosine similarity or euclidean distance for matching (most appropriate for semantic search).
                - k=5: Returns the 5 most similar chunks for each query - balances relevance with response time. '''
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            self.log.info(
                "3. [_create_retriever() ] in [ src\single_document_chat\data_ingestion.py ]. Retriever created successfully", 
                retriever_type=str(type(retriever)))
            return retriever  
        
        except Exception as e:
            self.log.error("3. [_create_retriever() ] in [ src\single_document_chat\data_ingestion.py ]. Retriever creation failed", error=str(e))
            raise DocumentPortalException("3. [_create_retriever() ] in [ src\single_document_chat\data_ingestion.py ]. Error creating FAISS retriever", sys)
    
    ''' *********************************************************************************************************************************************************************************************************************** '''