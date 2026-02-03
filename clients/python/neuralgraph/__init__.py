from .client import NGraphClient

__all__ = ["NGraphClient"]

# Optional: PyTorch Geometric utilities
try:
    from .torch_utils import to_pyg_data
    __all__.append("to_pyg_data")
except ImportError:
    pass

# Optional: LangChain integration
try:
    from .langchain_store import NeuralGraphStore
    from .chains import create_qa_chain, create_simple_chain, NeuralGraphQAChain
    __all__.extend([
        "NeuralGraphStore",
        "create_qa_chain",
        "create_simple_chain",
        "NeuralGraphQAChain",
    ])
except ImportError:
    pass
