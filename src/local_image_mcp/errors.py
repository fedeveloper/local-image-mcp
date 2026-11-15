"""Typed errors. Each message is written for the end user (a person or an MCP client)."""

from __future__ import annotations


class LocalImageError(Exception):
    """Base class for all errors that this package raises on purpose."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RegistryError(LocalImageError):
    """The model registry file is invalid."""


class UnknownModelError(LocalImageError):
    """The request names a model that is not in the registry."""


class CapabilityError(LocalImageError):
    """The request needs a capability that the model does not have."""


class InvalidRequestError(LocalImageError):
    """A request parameter is out of range or malformed."""


class MissingModelFilesError(LocalImageError):
    """One or more model files are not in the local cache."""


class BackendNotAvailableError(LocalImageError):
    """The backend is not installed, not configured or not implemented yet."""


class OutputPathError(LocalImageError):
    """The output path is not allowed or cannot be used."""


class JobNotFoundError(LocalImageError):
    """No job has this id."""


class JobCancelledError(LocalImageError):
    """The job was cancelled while it ran."""


class MetadataError(LocalImageError):
    """A metadata sidecar file is missing or invalid."""


class GenerationFailedError(LocalImageError):
    """The backend ran but did not produce an image."""
