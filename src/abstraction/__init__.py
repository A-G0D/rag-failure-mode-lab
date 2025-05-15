"""Abstraction layer: retriever and generator interfaces + implementations."""
from .retriever_interface import Retriever
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever
from .hybrid_retriever import HybridRetriever
from .generator_interface import Generator
from .mock_generator import ExtractiveGenerator
from .template_generator import TemplateGenerator

__all__ = [
    "Retriever", "BM25Retriever", "EmbeddingRetriever", "HybridRetriever",
    "Generator", "ExtractiveGenerator", "TemplateGenerator",
]
