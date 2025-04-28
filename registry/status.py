# --- registry/status.py ---
from enum import Enum

class ProcessingStatus(Enum):
    """Enum for document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def from_string(cls, status_str):
        """Convert a string to a ProcessingStatus enum."""
        for status in cls:
            if status.value == status_str:
                return status
        raise ValueError(f"Unknown status: {status_str}")