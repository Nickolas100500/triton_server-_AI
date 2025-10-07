"""
Utility modules for DialoGPT.

Keep this package import-safe. Optional submodules may not always be present
in every deployment, so avoid hard imports that break `from DialoGPT.utils.* import ...`.
"""

# Optional utilities. If unavailable, provide benign fallbacks.
try:
	from .data_loader import load_fallbacks, get_fallbacks_data  # type: ignore
except Exception:
	def load_fallbacks(*args, **kwargs):  # type: ignore
		"""Safe no-op fallback when data_loader is not present."""
		return {}

	def get_fallbacks_data(*args, **kwargs):  # type: ignore
		"""Safe no-op fallback when data_loader is not present."""
		return {}

# Export fallbacks provider class (optional)
try:
	from .fallbacks import DialoGPTFallbacks  # type: ignore
except Exception:
	DialoGPTFallbacks = None  # type: ignore

__all__ = [
	'load_fallbacks',
	'get_fallbacks_data',
	'DialoGPTFallbacks',
]