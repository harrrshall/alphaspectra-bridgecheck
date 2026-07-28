"""BandTrace deterministic model-sensor conformance preflight."""

from .audit import AuditResult, run_audit
from .authority import NormativeAuthority, verify_packaged_normative_authority
from .bundle import Bundle, load_bundle
from .constants import PRODUCT_VERSION
from .errors import BandTraceError, BundleError, ExecutionError
from .reference import make_reference_bundle

__all__ = [
    "AuditResult",
    "BandTraceError",
    "Bundle",
    "BundleError",
    "ExecutionError",
    "NormativeAuthority",
    "load_bundle",
    "make_reference_bundle",
    "run_audit",
    "verify_packaged_normative_authority",
]

__version__ = PRODUCT_VERSION
