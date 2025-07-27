import os
import utils.model_loader as ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentProcessingError
from model.models import *

from langchain_core.output_parsers import JSONOutputParser
from langchain.output_parsers import OutputFixingParser

class DocumentAnalyzer:

    def __init__(self):
        pass

    def analyze_document(self):
        pass