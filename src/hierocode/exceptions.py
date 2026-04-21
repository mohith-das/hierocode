class HierocodeError(Exception):
    """Base exception for all Hierocode errors."""
    pass

class ConfigError(HierocodeError):
    """Raised when there is a configuration issue."""
    pass

class ProviderConnectionError(HierocodeError):
    """Raised when a provider cannot be reached."""
    pass

class ModelNotFoundError(HierocodeError):
    """Raised when an expected model is missing on the provider."""
    pass
