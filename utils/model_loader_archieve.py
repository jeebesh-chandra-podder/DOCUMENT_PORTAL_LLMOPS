
import os
import sys

from dotenv import load_dotenv
from utils.config_loader import load_config

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

''' - The __name__ variable automatically gives the logger the name of the file (model_loader). 
    - This is a powerful convention for filtering logs from different parts of your application later on. '''
log = CustomLogger().get_logger(__name__)

''' - Defines a utility class for loading AI models.
    - Senior devs ask: "Do I need to maintain state between operations? Do related functions share common setup?" If yes → class. If no → standalone functions.'''
class ModelLoader:
    ''' *********************************************************************************************************************************************************************************************************************** '''
    def __init__(self):
        
        ''' - Loads key-value pairs from a .env file into the process environment. 
            - “Credentials live outside code.” A reflex that prevents accidental key leaks. '''
        load_dotenv()
        
        ''' - Immediately checks that required environment variables exist.
            - Don’t postpone validation—runtime errors are harder to trace. 
            - This is a private User Defined Function. '''
        self._validate_env()
        
        ''' - Loads the YAML configuration file.
            - The config file contains model parameters and other settings. '''
        self.config = load_config()
        
        log.info(
            "1. [ __init__() ]  in [ model_loader.py ] loaded successfully. Configuration loaded successfully", 
            config_keys=list(self.config.keys())
        )

    ''' *********************************************************************************************************************************************************************************************************************** '''
    ''' - Validate necessary environment variables.
        - Ensure API keys exist. '''
    def _validate_env(self):
        
        ''' - This line creates a list of strings. 
            - Each string is the exact name of an environment variable that the application must have to function correctly'''
        required_vars=[
            "GOOGLE_API_KEY",
            "GROQ_API_KEY"
        ]

        ''' - Dictionary comprehension that creates a mapping of variable names to their values.
            - It loops through your required_vars list and, for each variable name, 
                it tries to fetch the corresponding value from the system's environment variables using os.getenv(key)'''
        self.api_keys={
            key:os.getenv(key) for key in required_vars
        }

        ''' - List comprehension to find keys with missing (None/empty) values.
            - Separates the "finding missing" logic from the "handling missing" logic. Clean separation of concerns.
            - "I want to collect all problems first, then report them together. This gives better error messages than failing on the first missing key." '''
        missing = [
            k for k, v in self.api_keys.items() if not v
        ]

        ''' - If any variables are missing, logs the error with context and raises a custom exception.
            - The log includes both a human message and structured data (missing_vars=missing). This helps debugging and monitoring.'''
        if missing:           
            log.error(
                "2. [ _validate_env() ] in [ model_loader.py ]. Missing environment variables",
                missing_vars=missing
            )
            raise DocumentPortalException("Missing environment variables", sys)
        
        log.info(
            "1. [ _validate_env() ] in [ model_loader.py ]. Environment variables validated",
            available_keys=[k for k in self.api_keys if self.api_keys[k]]
        )
    ''' *********************************************************************************************************************************************************************************************************************** '''
    def load_embeddings(self):
        
        ''' - Loads an embedding model with error handling and logging.
            - Models can fail to load for various reasons (network, authentication, invalid model names). We want to catch these gracefully.
            - Why extract model_name first: Makes the code more readable and allows easier debugging - you can inspect the variable value.'''
        try:
            log.info("3. [ load_embeddings() ] in [ model_loader.py ]. Loading embedding model...")
            model_name = self.config["embedding_model"]["model_name"]
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except Exception as e:
            log.error("3. [ load_embeddings() ] in [ model_loader.py ]. Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    ''' *********************************************************************************************************************************************************************************************************************** '''
    
    ''' - Gets LLM configuration and determines which provider to use.
        '''
    def load_llm(self):
        
        llm_block = self.config["llm"]
        log.info("4. [ load_llm() ] in [ model_loader.py ]. Loading LLM...")
        
        ''' - Gets LLM configuration and determines which provider to use. 
            - "I want flexibility - users should be able to switch providers without code changes, but I need a reasonable default."'''
        provider_key = os.getenv(
            "LLM_PROVIDER", 
            "google"
        )  # Default groq
        
        
        ''' - Validates that the requested provider exists in configuration before proceeding.
            - Better to catch configuration errors immediately than let them cause mysterious failures later.'''
        if provider_key not in llm_block:
            log.error("4. [ load_llm() ] in [ model_loader.py ]. LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key}' not found in config")

        ''' - Extracts configuration values with defaults for optional parameters. 
            - Why .get() with defaults: dict.get(key, default) is safer than dict[key] - it won't crash if the key is missing.
            - Senior Thinking: "Temperature and max_tokens are reasonable to have defaults for, but provider and model_name are required. I'll use .get() for optional params only."'''
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)
        
        log.info(
            "4. [ load_llm() ] in [ model_loader.py ]. Loading LLM",
            provider=provider,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        ''' - Implements a factory pattern - different object creation based on input parameters.
            - Allows adding new providers without changing the interface. Each provider has different initialization requirements.
            - "I want one method that can create different types of LLMs. The caller shouldn't need to know the specifics of each provider."'''
        if provider == "google":
            llm=ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            return llm
        elif provider == "groq":
            llm=ChatGroq(
                model=model_name,
                api_key=self.api_keys["GROQ_API_KEY"],
                temperature=temperature,
            )
            return llm            
        elif provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_keys["OPENAI_API_KEY"],
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            log.error("❌[ load_llm() ] in [ model_loader.py ]. Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
    
''' *********************************************************************************************************************************************************************************************************************** '''   

''' - Provides a simple test when the file is run directly.
    - Why if __name__ == "__main__": Allows the file to be imported as a module without running the test code.
    - Senior Habit: "I always include basic smoke tests. Future me will thank me when I need to quickly verify the module works."
    - Mental Model: Think of this as "developer documentation through examples" - shows how to use your class. '''
if __name__ == "__main__":
    loader = ModelLoader()
    
    # Test embedding model loading
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    
    # Test LLM loading based on YAML config
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    
    # Test the ModelLoader
    result=llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")