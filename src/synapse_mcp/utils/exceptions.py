"""
Custom exceptions for Project Synapse.
"""


class SynapseError(Exception):
    """Base class for all Synapse exceptions."""


class WikiError(SynapseError):
    """Base class for Wiki-related errors."""


class WikiPageNotFoundError(WikiError):
    """Raised when a requested wiki page does not exist."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Wiki page not found: {path}")


class WikiAccessError(WikiError):
    """Raised when a wiki page cannot be accessed (permissions, lock, etc.)."""


class WikiIndexError(WikiError):
    """Raised when there is an issue with the wiki index."""
