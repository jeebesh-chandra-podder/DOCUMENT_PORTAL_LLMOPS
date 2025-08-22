import uuid
from pathlib import Path
import sys
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader

class DocumentIngestor:
    SUPPORTED_FILE_TYPES = {'.pdf','.docx','.txt','.md'}
    
    ''' *********************************************************************************************************************************************************************************************************************** '''
    def __init__(self, temp_dir:str = 'data\mulit_doc_chat', faiss_dir:str = 'faiss_index', session_id:str | None = None):
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
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            '''
             - Builds a unique folder name e.g., session_20250729_053001_a1b2c3d4.
             - Combines timestamp (sortable) and shortened UUID (collision-proof).
             - The hex[:8] slice shortens 32-char UUID to 8, keeps paths concise
            '''
            self.session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_temp_dir = self.temp_dir / self.session_id
            self.session_faiss_dir = self.faiss_dir / self.session_id
            self.session_temp_dir.mkdir(parents = True, exist_ok = True)
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()

            self.log.info(
                "1. [__init__() ] in [ src\multi_document_chat\data_ingestion.py ].  Document Ingestor Initialized",
                temp_base = str(self.temp_dir),
                faiss_base = str(self.faiss_dir),
                session_id = self.session_id,
                temp_path = str(self.session_temp_dir),
                faiss_path = str(self.session_faiss_dir)
            )

        except Exception as e:
            self.log.error("1. [__init__() ] in [ src\multi_document_chat\data_ingestion.py ]. Failed to initialize DocumentIngestor", error=str(e))
            raise DocumentPortalException("1. [__init__() ] in [ src\multi_document_chat\data_ingestion.py ]. Initialization error in DocumentIngestor", sys)

    def ingest_files(self, uploaded_files):
        try:
            documents = []
            for uploaded_file in uploaded_files:
                ext = Path(uploaded_file.name).suffix.lower()
                if ext not in self.SUPPORTED_FILE_TYPES:
                    self.log.warning("Unsupported File Skipped", filename = uploaded_file.name)
                    continue
                unique_filename = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                temp_path = self.session_temp_dir / unique_filename

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                self.log.info(
                    "1. [ingest_files() ] in [ src\multi_document_chat\data_ingestion.py ]. File Saved for Ingestion",
                    filename = uploaded_file.name,
                    saved_as = str(temp_path),
                    session_id = self.session_id
                )

                if ext == '.pdf':
                    loader = PyPDFLoader(str(temp_path))

                elif ext == '.docx':
                    loader = Docx2txtLoader(str(temp_path))
                
                elif ext == '.txt':
                    loader = TextLoader(str(temp_path), encoding="utf-8")
                
                else:
                    self.log.warning("Unsupported File Skipped", filename = uploaded_file.name)
                    continue

                docs = loader.load()
                documents.extend(docs)

                if not documents:
                    raise DocumentPortalException("1. [ingest_files() ] in [ src\multi_document_chat\data_ingestion.py ]. No Walid Documents Loaded", sys)
                
                self.log.info(
                    "1. [ingest_files() ] in [ src\multi_document_chat\data_ingestion.py ]. All Documents Loaded",
                    total_docs = len(documents),
                    session_id = self.session_id
                )

                return self._create_retriever(documents)

        except Exception as e:
            self.log.error("2. [ingest_files() ] in [ src\multi_document_chat\data_ingestion.py ]. Failed to initialize DocumentIngestor", error=str(e))
            raise DocumentPortalException("2. [ingest_files() ] in [ src\multi_document_chat\data_ingestion.py ]. Initialization error in DocumentIngestor", sys)

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 300
            )

            chunks = splitter.split_documents(documents)

            self.log.info(
                "3. [_create_retriever() ] in [ src\multi_document_chat\data_ingestion.py ]. Documents SPlit Into Chunks",
                total_chunks = len(chunks),
                session_id = self.session_id
            )

            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vectorstore.save_local(str(self.faiss_dir))
            self.log.info(
                "3. [_create_retriever() ] in [ src\multi_document_chat\data_ingestion.py ]. FAISS index created and saved", 
                faiss_path=str(self.faiss_dir)
            )
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            self.log.info(
                "3. [_create_retriever() ] in [ src\multi_document_chat\data_ingestion.py ]. Retriever created successfully", 
                retriever_type=str(type(retriever)))
            return retriever 

        except Exception as e:
            self.log.error("1. [__init__() ] in [ src\multi_document_chat\data_ingestion.py ]. Failed to initialize DocumentIngestor", error=str(e))
            raise DocumentPortalException("1. [__init__() ] in [ src\multi_document_chat\data_ingestion.py ]. Initialization error in DocumentIngestor", sys)

