"""Stable public exception types."""


class BandTraceError(Exception):
    """Base class for expected BandTrace failures."""


class BundleError(BandTraceError):
    """The submitted bundle is invalid or unsafe to inspect."""


class ExecutionError(BandTraceError):
    """The declared executable could not produce a valid bounded result."""

