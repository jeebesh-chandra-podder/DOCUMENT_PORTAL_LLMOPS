import os
import fitz # PyMuPDF to open and read PDF files
import uuid # Provides universally unique identifiers. Session folders must not collide; UUID slices guarantee uniqueness even if two users start jobs in the same millisecond.
from datetime import datetime
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

class DocumentHandler:
    """
    Handles PDF saving and reading operations.
    Automatically logs all actions and supports session-based organization.
    """

    ''' *********************************************************************************************************************************************************************************************************************** '''
    def __init__(self, data_dir=None, session_id=None):
        try:
            
            ''' Sets up an instance logger tagged with this module name. '''
            self.log = CustomLogger().get_logger(__name__)

            ''' Sets the data directory to store PDFs.
                Defaults to an environment variable or a subdirectory in the current working directory. '''
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "data", "document_analysis")
            )
            
            '''
             - Builds a unique folder name e.g., session_20250729_053001_a1b2c3d4.
             - Combines timestamp (sortable) and shortened UUID (collision-proof).
             - The hex[:8] slice shortens 32-char UUID to 8, keeps paths concise
            '''
            self.session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            ''' Constructs and creates the session directory. '''
            self.session_path = os.path.join(
                self.data_dir, 
                self.session_id
            )
            os.makedirs(
                self.session_path, 
                exist_ok=True
            )

            self.log.info(
                "__init__() called Successfully. PDFHandler initialized. Successfully created session directory",
                session_id=self.session_id,
                session_path=self.session_path
            )

        except Exception as e:
            self.log.error(f"Error in initializing DocumentHandler: {e}")
            raise DocumentPortalException("Error in initializing DocumentHandler", e) from e

    ''' *********************************************************************************************************************************************************************************************************************** '''
    def save_pdf(self, uploaded_file):
        try:

            ''' - Grabs only the file’s base name, stripping directories browsers sometimes include.
                - Prevents directory traversal attacks (../../etc/passwd).
                - Never trust user paths; sanitize ASAP. '''
            filename = os.path.basename(uploaded_file.name)

            ''' - Simple extension validation. 
                - Avoids accidental .docx uploads that would break PyMuPDF later. '''
            if not filename.lower().endswith(".pdf"):
                raise DocumentPortalException("Invalid file type. Only PDFs are allowed.")

            ''' - Destination path under the session directory.
                - Keeps user-supplied names but within a controlled folder.
                - Containment principle – never scatter user files across disk. '''
            save_path = os.path.join(self.session_path, filename)

            ''' - Streams the uploaded binary to disk.
                - "wb" ensures raw bytes saved; some frameworks deliver BytesIO.
                - Use context managers (with) so file handles always close, even on error.'''
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            self.log.info(
                "save_pdf() called Successfully. PDF saved successfully", 
                file=filename, 
                save_path=save_path, 
                session_id=self.session_id
            )
            
            return save_path
        
        except Exception as e:
            self.log.error(f"Error saving PDF: {e}")
            raise DocumentPortalException("Error saving PDF", e) from e

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' Declares a method that takes a file path to a PDF and promises to return a string.'''
    def read_pdf(self, pdf_path: str) -> str:
        try:
            text_chunks = []
            
            ''' - Opens the PDF in a context manager so the file handler auto-closes.
                - PyMuPDF (fitz) is fast for text extraction and the with block protects against leaked OS file descriptors.
                - Seniors mentally tag any “open something” with a context manager—file, DB, network socket—then forget about manual cleanup. '''
            with fitz.open(pdf_path) as doc:

                ''' - Loops through every page, yielding both the page object and a 1-based index.
                    - Human-friendly numbering (start=1) matches what readers see in Acrobat; avoids the off-by-one headache of zero indexing. 
                    - Enumerate when you need both index and item; otherwise use a direct for page in doc. '''
                for page_num, page in enumerate(doc, start=1):
                    
                    ''' - Extracts text from the current page and stores it with a page-header separator. 
                        - Invest a tiny formatting cost now to save massive debugging time later when a user asks, “Which page did this answer come from?” '''
                    text_chunks.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
            
            ''' - Stitches every page string into one large text body separated by newlines. 
                - Single return value is easier for callers; they can split on '--- Page' if needed. '''
            text = "\n".join(text_chunks)

            self.log.info(
                "read_pdf() called Successfully. PDF read successfully", 
                pdf_path=pdf_path, 
                session_id=self.session_id, 
                pages=len(text_chunks)
            )

            return text
        
        except Exception as e:
            self.log.error(f"Error reading PDF: {e}")
            raise DocumentPortalException("Error reading PDF", e) from e

''' *********************************************************************************************************************************************************************************************************************** '''    
''' - Executes the indented code only when the file is run directly, not when it’s imported. 
    - Keeps your module usable both as an importable library (production) and a quick CLI test tool (development).'''
if __name__ == "__main__":
    from pathlib import Path # Path makes file-name manipulations (.name, .stem, .suffix) more readable than os.path.
    from io import BytesIO # Pulls in an in-memory binary stream class.

    pdf_path=r"C:\\Projects\\LLMOPS\\DOCUMENT_PORTAl\\data\\document_analysis\\NIPS-2017-attention-is-all-you-need-Paper.pdf"
    class DummyFile:
        
        ''' - Stores the file name separately (mimicking uploaded_file.name) and keeps the path private. 
            - save_pdf() calls os.path.basename(uploaded_file.name)—the class must supply exactly that attribute. '''
        def __init__(self,file_path):
            self.name = Path(file_path).name
            self._file_path = file_path
        
        ''' - Reads the entire PDF into memory and returns the bytes. 
            - getbuffer() mimics the BytesIO interface, allowing save_pdf() to call uploaded '''
        def getbuffer(self):
            return open(self._file_path, "rb").read()

    dummy_pdf = DummyFile(pdf_path)

    ''' - Builds the main service object using default settings (env var or local ./data/document_analysis). 
        - Mirrors how production code will call it—good smoke test. '''
    handler = DocumentHandler()
    
    try:
        
        ''' - Pipes the dummy file through the real save logic and captures its final location. 
            - Validates path creation, extension check, UUID folder, actual disk write—all in one call.
            - One function call exercises half the codebase; embrace high-value integration tests.'''
        saved_path=handler.save_pdf(dummy_pdf)
        print(saved_path)
        
        ''' - Feeds the just-saved PDF back into read_pdf() to extract text. 
            - End-to-end check that both write and read pipelines work consistently.
            - Round-trip tests (write → read) surface path mistakes, encoding issues, and permission errors in one go. '''
        content = handler.read_pdf(saved_path)
        
        print("PDF Content:")
        print(content[:500])  # Print first 500 characters of the PDF content
        
    except Exception as e:
        print(f"Error: {e}")
    
    