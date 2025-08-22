import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY # type: ignore

class DocumentAnalyzer:
    
    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Analyzes documents using a pre-trained model.
        - Automatically logs all actions and supports session-based organization. '''
    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader=ModelLoader()
            
            ''' - Acquire the actual LLM client from the loader.
                - Enables swapping providers (OpenAI, local, Azure) without touching the analyzer. '''
            self.llm=self.loader.load_llm()

            ''' - A JsonOutputParser configured with a Pydantic model called Metadata. 
                - It enforces that the LLM’s response matches a specific JSON schema (the fields/types defined by Metadata), then returns a typed object/dict instead of a free-form string.'''
            self.parser = JsonOutputParser(pydantic_object=Metadata)

            ''' - A wrapper around self.parser that can ask the LLM to repair malformed outputs so they conform to the Metadata schema. 
                - Even with instructions, LLMs can produce almost-correct JSON (extra text, trailing commas, missing fields). The fixing parser automatically attempts a “self-heal” pass rather than hard-failing.'''
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            ''' - Load the document analysis prompt template from a central registry. 
                - Treat prompts as config, not inline strings. Centralization enables governance and rapid iteration. '''
            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("1. [ __init__() ] from [ src\document_analyzer\data_analysis.py ]. DocumentAnalyzer initialized successfully")


        except Exception as e:
            self.log.error(f"1. [ __init__() ] from [ src\document_analyzer\data_analysis.py ]. Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException("1. [ __init__() ] from [ src\document_analyzer\data_analysis.py ]. Error in DocumentAnalyzer initialization", sys)
        
        
    ''' *********************************************************************************************************************************************************************************************************************** '''
    def analyze_document(self, document_text:str)-> dict:
        ''' - Analyze a document's text and extract structured metadata & summary. '''
        try:
            ''' - Compose a LangChain-style pipeline using operator overloading:
                    1. self.prompt: a prompt template that formats instructions and inputs.
                    2. self.llm: the language model that generates output.
                    3. self.fixing_parser: a parser that validates and, if needed, repairs output to match the schema.
                - This creates a declarative, readable “flow”: Prompt → LLM → Parser. '''
            chain = self.prompt | self.llm | self.fixing_parser
            
            self.log.info("2. [ analyze_document() ] from [ src\document_analyzer\data_analysis.py ]. Meta-data analysis chain initialized")

            ''' - What: 
                    - Calls the chain with a dict of variables the prompt expects.
                    - format_instructions is pulled from the parser to instruct the LLM how to format output so it can be parsed reliably.
                    - document_text is the user-provided content to analyze.
                - Why:
                    - Injecting format_instructions directly from the parser closes the loop between what the parser enforces and what the LLM is asked to produce—reduces mismatch errors.
                    - Passing document_text keeps the prompt template generic and reusable. '''
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            self.log.info("2. [ analyze_document() ] from [ src\document_analyzer\data_analysis.py ]. Metadata extraction successful", keys=list(response.keys()))
            
            return response

        except Exception as e:
            self.log.error("2. [ analyze_document() ] from [ src\document_analyzer\data_analysis.py ]. Metadata analysis failed", error=str(e))
            raise DocumentPortalException("2. [ analyze_document() ] from [ src\document_analyzer\data_analysis.py ]. Metadata extraction failed",sys)
    
    ''' *********************************************************************************************************************************************************************************************************************** '''
        
    