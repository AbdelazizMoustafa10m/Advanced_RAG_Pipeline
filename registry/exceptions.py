# --- registry/exceptions.py ---

class DocumentRegistryError(Exception):
    """Base exception for all document registry errors."""
    pass

class DocumentNotFoundError(DocumentRegistryError):
    """Raised when a document is not found in the registry."""
    pass

class DocumentAlreadyExistsError(DocumentRegistryError):
    """Raised when attempting to add a document that already exists."""
    pass

class RegistryConnectionError(DocumentRegistryError):
    """Raised when there is an error connecting to the registry database."""
    pass

class InvalidStatusError(DocumentRegistryError):
    """Raised when an invalid status is provided."""
    pass

class InvalidDocumentIdError(DocumentRegistryError):
    """Raised when an invalid document ID is provided."""
    pass