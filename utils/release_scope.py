"""Exceptions for components outside the scope of this source distribution."""


class ComponentUnavailableError(RuntimeError):
    """Raised when an unavailable component is invoked."""
