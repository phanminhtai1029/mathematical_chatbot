# Simple configuration file

# Model settings
DEFAULT_MODEL = "mathstral"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# LLM settings
TEMPERATURE = 0.1
MAX_TOKENS = 2048

# GPU settings
USE_GPU = True  # Set to False if you want CPU only
GPU_DEVICE = "cuda"  # or "cuda:0" for specific GPU

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
DEFAULT_PDF = "MAS-Douglas-C.-Montgomery-267-298.pdf"
DEFAULT_VECTORS = "vector-store_MAS-Douglas-C.-Montgomery-267-298.pdf"