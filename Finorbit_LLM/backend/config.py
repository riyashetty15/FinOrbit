"""
Configuration management for unified backend

This module provides centralized configuration using Pydantic Settings,
replacing scattered environment variable access throughout the codebase.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Centralized configuration for the unified backend.

    All settings are loaded from environment variables (.env file).
    This replaces the previous pattern of os.getenv() calls scattered
    throughout backend/server.py and backend/agents.py.
    """

    # ==================== API Configuration ====================
    base_url: str = "https://api.openai.com/v1/"
    llm_api_key: str
    custom_model_name: str = "gpt-4o-mini"
    header_id: Optional[str] = None

    # ==================== LLM Provider ====================
    llm_provider: str = "openai"        # "openai" | "gemini"
    google_api_key: Optional[str] = None

    # ==================== Database ====================
    database_url: str

    # ==================== Validation Thresholds ====================
    confidence_threshold: float = 0.7
    pii_block_critical: bool = True
    content_risk_block_critical: bool = True

    # ==================== Logging ====================
    log_level: str = "INFO"
    log_file: str = "backend.log"
    json_logs: bool = True

    # ==================== Agent Settings ====================
    max_agent_turns: int = 30
    agent_timeout_seconds: int = 60
    use_custom_models: bool = True

    # ==================== Feature Flags ====================
    # Enable/disable components for gradual rollout
    validation_pipeline_enabled: bool = True
    enforce_guardrails: bool = True  # Set False to revert to analysis-only mode

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Allow lowercase env vars
        extra = "ignore"  # Ignore extra env vars


# Global settings instance
settings = Settings()
