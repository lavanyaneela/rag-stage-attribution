"""Stage attribution for retrieval-augmented pipelines.

A wrong answer from a RAG pipeline cannot say, by itself, whether the needed
chunk was never retrieved or was retrieved and ignored. This package measures
how well a trace-based attribution method recovers the true failing stage,
against failures it injected and therefore knows the truth about.
"""

__version__ = "1.0.0"
