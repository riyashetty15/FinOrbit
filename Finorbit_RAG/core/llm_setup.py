"""
LLM Configuration and Setup.

Handles the initialization of the Global LlamaIndex Settings with the configured LLM provider
(OpenAI, Gemini, etc.).
"""

import logging
import os
from llama_index.core import Settings
from config import get_llamaindex_config

logger = logging.getLogger(__name__)

def configure_global_llm():
    """
    Configure the global LLM for LlamaIndex based on configuration.
    
    Supported providers:
    - openai (default)
    - gemini
    """
    config = get_llamaindex_config()
    
    provider = config.llm_provider
    model = config.llm_model
    
    logger.info(f"Configuring LLM provider: {provider}")
    
    try:
        if provider == "gemini":
            from llama_index.llms.gemini import Gemini
            
            if not config.google_api_key:
                logger.warning("GOOGLE_API_KEY not found. Gemini may fail to initialize.")
            
            # Default to gemini-pro if not specified
            model_name = model or "models/gemini-1.5-flash"
            
            Settings.llm = Gemini(
                model=model_name,
                api_key=config.google_api_key,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            logger.info(f"Set global LLM to Gemini: {model_name}")
            
        elif provider == "openai":
            # Accessing openai (assuming it is installed or will be if used)
            try:
                from llama_index.llms.openai import OpenAI
                
                # Default to gpt-3.5-turbo if not specified
                model_name = model or "gpt-3.5-turbo"
                
                Settings.llm = OpenAI(
                    model=model_name,
                    temperature=config.llm_temperature,
                    max_tokens=config.llm_max_tokens
                )
                logger.info(f"Set global LLM to OpenAI: {model_name}")
            except ImportError:
                 logger.warning("llama-index-llms-openai not installed. Skipping OpenAI setup.")

        else:
            logger.warning(f"Unknown LLM provider: {provider}. Leaving default settings.")
            
    except Exception as e:
        logger.error(f"Failed to configure LLM: {e}")
        # Build logic to handle failure or fallback? 
        # For now, just log error.
