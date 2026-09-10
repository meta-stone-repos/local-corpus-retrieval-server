from __future__ import annotations

from .base import Provider, check_keys
from .zhipu import ZhipuProvider
from .iqs import IQSProvider


PROVIDERS = {
    "iqs": IQSProvider,
    "zhipu": ZhipuProvider,
}


def create_provider(config: dict) -> Provider:
    check_keys(config, {"provider", "providers"}, "root")
    name = config.get("provider")
    if not isinstance(name, str) or name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name!r}")

    sections = config.get("providers")
    if not isinstance(sections, dict):
        raise ValueError("providers must be an object")
    options = sections.get(name)
    if not isinstance(options, dict):
        raise ValueError(f"Missing configuration for provider: {name}")
    return PROVIDERS[name](options)
