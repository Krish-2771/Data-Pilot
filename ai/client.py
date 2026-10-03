"""
NVIDIA NIM / OpenAI-compatible client for Data-Pilot AI module.

Uses environment variables for configuration:
- NVIDIA_API_KEY: Required API key (never hard-coded)
- NVIDIA_NIM_BASE_URL: Base URL for NIM API (default: https://integrate.api.nvidia.com/v1)
- NVIDIA_MODEL: Model name (default: nvidia/nemotron-3.5-lightning-30b-a3b)
- NVIDIA_TEMPERATURE: Temperature for generation (default: 0.2)
- NVIDIA_MAX_TOKENS: Max tokens for response (default: 2000)
"""

import os
from typing import Optional
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from ai.exceptions import (
    ConfigurationError,
    MissingAPIKeyError,
    APIError,
    APITimeoutError,
)
from ai.schemas import AISystemPromptConfig


# Load environment variables from .env file
load_dotenv()


class NIMClient:
    """
    NVIDIA NIM client using OpenAI-compatible API.

    Provides both synchronous and asynchronous interfaces for
    generating structured recommendations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the NIM client.

        Args:
            api_key: NVIDIA API key. If None, reads from NVIDIA_API_KEY env var.
            base_url: NIM base URL. If None, reads from NVIDIA_NIM_BASE_URL env var.
            model: Model name. If None, reads from NVIDIA_MODEL env var.
            temperature: Temperature for generation. If None, reads from NVIDIA_TEMPERATURE env var.
            max_tokens: Max tokens for response. If None, reads from NVIDIA_MAX_TOKENS env var.
            timeout: Request timeout in seconds.

        Raises:
            MissingAPIKeyError: If no API key is provided or found in environment.
            ConfigurationError: If configuration is invalid.
        """
        # Resolve configuration from parameters or environment
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = base_url or os.getenv(
            "NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self.model = model or os.getenv(
            "NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
        )

        try:
            self.temperature = (
                temperature
                if temperature is not None
                else float(os.getenv("NVIDIA_TEMPERATURE", "0.2"))
            )
            self.max_tokens = (
                max_tokens
                if max_tokens is not None
                else int(os.getenv("NVIDIA_MAX_TOKENS", "2000"))
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid numeric configuration: {e}")

        self.timeout = timeout

        # Validate required configuration
        if not self.api_key:
            raise MissingAPIKeyError(
                "NVIDIA API key not configured. "
                "Set NVIDIA_API_KEY environment variable or pass api_key parameter."
            )

        # Create OpenAI-compatible clients
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )
        self._async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    @classmethod
    def from_env(cls) -> "NIMClient":
        """Create a client from environment variables or Streamlit secrets."""
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                from streamlit.errors import StreamlitSecretNotFoundError

                api_key = st.secrets.get("NVIDIA_API_KEY")
            except StreamlitSecretNotFoundError:
                # Local runs without a secrets file should still start; AI use
                # will raise MissingAPIKeyError with a clear configuration message.
                api_key = None
        return cls(api_key=api_key)

    def get_config(self) -> AISystemPromptConfig:
        """Return current configuration as schema."""
        return AISystemPromptConfig(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
        )

    def chat_completion(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: bool = False,
        reasoning_budget: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to NIM.

        Args:
            messages: List of message dictionaries (system, user, assistant).
            response_format: Optional response format for structured output.
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.
            enable_thinking: Enable nemotron reasoning mode (adds reasoning_content).
            reasoning_budget: Max tokens for reasoning (nemotron specific).

        Returns:
            The model's response content as string (final answer, not reasoning).

        Raises:
            APIError: If API call fails.
            APITimeoutError: If request times out.
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format

            # Nemotron-specific: disable thinking for JSON output
            # Always disable thinking for nemotron when response_format is json_object
            # or when explicitly requested
            if ("nemotron" in self.model.lower() and
                (response_format is not None or enable_thinking is False)):
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if reasoning_budget:
                    kwargs["extra_body"]["reasoning_budget"] = reasoning_budget

            response = self._client.chat.completions.create(**kwargs)

            # For nemotron, the JSON is in content, reasoning is in reasoning_content
            content = response.choices[0].message.content
            if content is None:
                # Check if there's reasoning content instead
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
                if reasoning:
                    return reasoning.strip()
                raise APIError("Empty response from NIM")

            return content.strip()

        except Exception as e:
            if "timeout" in str(e).lower():
                raise APITimeoutError(f"NIM request timed out: {e}")
            raise APIError(f"NIM API error: {e}")

    async def achat_completion(
        self,
        messages: list[dict],
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send an async chat completion request to NIM.

        Args:
            messages: List of message dictionaries.
            response_format: Optional response format for structured output.
            temperature: Optional temperature override.
            max_tokens: Optional max_tokens override.

        Returns:
            The model's response content as string.

        Raises:
            APIError: If API call fails.
            APITimeoutError: If request times out.
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = await self._async_client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            if content is None:
                raise APIError("Empty response from NIM")

            return content.strip()

        except Exception as e:
            if "timeout" in str(e).lower():
                raise APITimeoutError(f"NIM request timed out: {e}")
            raise APIError(f"NIM API error: {e}")

    def test_connection(self) -> bool:
        """
        Test connection to NIM with a minimal request.

        Returns:
            True if connection successful, False otherwise.

        Note:
            Does not raise exceptions; returns False on any error.
        """
        try:
            response = self.chat_completion(
                messages=[
                    {"role": "user", "content": "ping"}
                ],
                max_tokens=10,
            )
            return True
        except Exception:
            return False
