"""
Custom exceptions for the Data-Pilot AI module.
"""


class DataPilotAIError(Exception):
    """Base exception for Data-Pilot AI errors."""
    pass


class ConfigurationError(DataPilotAIError):
    """Raised when AI configuration is invalid or missing."""
    pass


class MissingAPIKeyError(ConfigurationError):
    """Raised when NVIDIA API key is not configured."""
    pass


class InvalidModelError(ConfigurationError):
    """Raised when model configuration is invalid."""
    pass


class APIError(DataPilotAIError):
    """Raised when NVIDIA NIM API call fails."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class APITimeoutError(APIError):
    """Raised when NVIDIA NIM API request times out."""
    pass


class MalformedResponseError(DataPilotAIError):
    """Raised when AI response cannot be parsed."""

    def __init__(self, message: str, raw_response: str = None):
        super().__init__(message)
        self.raw_response = raw_response


class ValidationError(DataPilotAIError):
    """Raised when AI recommendation fails validation."""
    pass


class UnsupportedActionError(ValidationError):
    """Raised when AI recommends an unsupported preprocessing action."""
    pass


class InvalidColumnError(ValidationError):
    """Raised when AI recommends action on non-existent column."""
    pass


class InvalidConfidenceError(ValidationError):
    """Raised when AI recommendation confidence is invalid."""
    pass