"""Boosted Higgs fatjet tagger studies."""

from typing import Any


def main(*args: Any, **kwargs: Any) -> Any:
    from .cli import main as cli_main

    return cli_main(*args, **kwargs)

__all__ = ["main"]
