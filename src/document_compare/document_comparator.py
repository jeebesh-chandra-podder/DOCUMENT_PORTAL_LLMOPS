import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse,PromptType

class DocumentComparatorLLM:
    ''' *********************************************************************************************************************************************************************************************************************** '''
    def __init__(self):
        load_dotenv()
        self.log = CustomLogger().get_logger(__name__)
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        
        ''' - Defines a strict parser that validates/casts the model’s output into a Pydantic schema (SummaryResponse). 
            - LLMs output text; downstream needs structure. A schema is a contract that makes errors explicit and data dependable.'''
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        
        ''' - Builds a wrapper that can auto-repair malformed outputs by asking the LLM to fix them to the schema. 
            - Real-world LLMs occasionally violate formatting; the fixer increases robustness. '''
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        
        ''' - Retrieves a reusable prompt template for document comparison from a central registry. 
            - Prompts evolve fast; a registry makes them addressable, versionable, testable, and swappable. '''
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        
        ''' - Assembles a linear pipeline: render prompt → call LLM → parse JSON into Python structures. 
            - LCEL-style composition clarifies data flow and separates concerns. The chain will accept inputs matching the prompt’s variables. '''
        self.chain = self.prompt | self.llm | self.parser

        self.log.info("1. [ __init__() ] from [ src\\document_compare\\document_comparator.py ]. DocumentComparatorLLM initialized", model=self.llm)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Declares a public API that takes a single string containing the contents of documents to compare, and returns a pandas DataFrame with structured comparison results. '''
    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            
            ''' - Prepares the variable dict for the prompt template. 
                    - combined_docs: Supplies the content to analyze.
                    - format_instruction: Injects the parser’s format instructions so the LLM emits output that matches the Pydantic schema (SummaryResponse).'''
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }

            self.log.info("2. [ compare_documents() ] from [ src\\document_compare\\document_comparator.py ]. Invoking document comparison LLM chain")

            ''' - Runs the end-to-end pipeline: prompt → LLM → parser. 
                - Because the chain ends with a parser, response should already be parsed Python (e.g., list[dict] shaped according to SummaryResponse). '''
            response = self.chain.invoke(inputs)
            self.log.info("2. [ compare_documents() ] from [ src\\document_compare\\document_comparator.py ]. Chain invoked successfully", response_preview=str(response)[:200])
            return self._format_response(response)
        except Exception as e:
            self.log.error("2. [ compare_documents() ] from [ src\\document_compare\\document_comparator.py ]. Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame: #type: ignore
        try:
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            self.log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)
    ''' *********************************************************************************************************************************************************************************************************************** '''