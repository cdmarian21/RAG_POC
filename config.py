import os

PDF_DIRECTORY = "./data" 
DB_PATH = "./faiss_db"

SLM_MODEL = "gemma4:e2b"  
EMBEDDING_MODEL = "mxbai-embed-large"

FAISS_DIMENSION = 1024  # mxbai-embed-large specs
FAISS_NLIST = 100       # this is set for 10,000 chunks change if necessary
FAISS_M = 32            # change based off embedding model specs
FAISS_NBITS = 8         # for my precious RAM
NPROB = 20              # number of neigbors (voroni cells) to search 
KWARGS = 8              # amount of relevant results

CHUNK_SIZE = 1000       # based off use case and slm model
CHUNK_OVERLAP = 200     # prevents cutting off info in chunks 