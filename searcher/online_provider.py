"""Compatibility imports for existing online entry points."""

if __package__:
    from .providers.base import Provider, check_keys, valid_url
    from .providers.registry import PROVIDERS, create_provider
    from .providers.zhipu import ZhipuProvider
else:
    from providers.base import Provider, check_keys, valid_url
    from providers.registry import PROVIDERS, create_provider
    from providers.zhipu import ZhipuProvider

__all__ = [
    "Provider", "check_keys", "valid_url",
    "PROVIDERS", "create_provider", "ZhipuProvider",
]
