"""Utility modules for CS4."""

from cs4.utils.config_loader import load_yaml, stamp, fill_vars
from cs4.utils.io_utils import ensure_dir
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.utils.llm_client import OpenAIClient, AnthropicClient, get_total_usage
from cs4.utils.embedding_utils import (
    load_or_create_embeddings,
    find_dissimilar_pairs,
    find_dissimilar_pairs_distinct,
    save_pairs_to_csv
)

__all__ = [
    "load_yaml",
    "stamp",
    "fill_vars",
    "ensure_dir",
    "setup_logging",
    "get_logger",
    "OpenAIClient",
    "AnthropicClient",
    "get_total_usage",
    "load_or_create_embeddings",
    "find_dissimilar_pairs",
    "find_dissimilar_pairs_distinct",
    "save_pairs_to_csv",
]
