"""whisper-omega package."""

from whisper_omega.api import transcribe_file
from whisper_omega.runtime.models import TranscriptionResult
from whisper_omega.runtime.policy import PolicyConfig
from whisper_omega.runtime.service import ServiceConfig, TranscriptionService

__all__ = [
    "__version__",
    "PolicyConfig",
    "ServiceConfig",
    "TranscriptionResult",
    "TranscriptionService",
    "transcribe_file",
]

__version__ = "0.1.0"
