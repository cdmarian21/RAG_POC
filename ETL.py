import os
import uuid
import numpy as np
import faiss
import pymupdf4llm
from markitdown import MarkItDown

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

import config
from chunking import chunk_markdown_document

def load_and_convert_pdfs_to_markdown(directory):
    markdown_docs = []
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    md_converter = MarkItDown()         # converter for the word docs
        
    # convert from pdf to markdown for better accuracy
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        md_text = ""
        
        try:
            # for pdfs
            if filename.lower().endswith(".pdf"):
                print(f"Processing PDF: {filename}")
                md_text = pymupdf4llm.to_markdown(filepath)
                
            # for word docs
            elif filename.lower().endswith((".docx", ".doc")):
                print(f"Processing Word Doc: {filename}")
                result = md_converter.convert(filepath)
                md_text = result.text_content
                
            else:
                # skip any others
                continue
                
            # add source metadata for better accuracy
            if md_text:
                markdown_docs.append({"text": md_text, "source": filename})
                
        except Exception as e:
            print(f"Error for {filename}: {e}")         # catch any misc errors
            
    return markdown_docs

def build_and_save_index():
    raw_md_docs = load_and_convert_pdfs_to_markdown(config.PDF_DIRECTORY)
    
    if not raw_md_docs:
        print("No documents found in the data directory")
        return

    # chunking the markdown file and saving the source for SLM reference
    all_chunks = []
    for doc in raw_md_docs:
        chunks = chunk_markdown_document(doc["text"])
        for chunk in chunks:
            chunk.metadata["source"] = doc["source"]
        all_chunks.extend(chunks)

    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    if len(all_chunks) < 2000:
        # basic for smaller data files
        faiss_index = faiss.IndexFlatL2(config.FAISS_DIMENSION)
    else:
        # if the input files is over 2000 chunks then we will use IVFPQ for speed
        # untested and probably wrong!
        sample_size = min(4000, len(all_chunks))
        sample_texts = [chunk.page_content for chunk in all_chunks[:sample_size]]
        sample_embeddings = embeddings.embed_documents(sample_texts)
        train_matrix = np.array(sample_embeddings).astype('float32')
        
        quantizer = faiss.IndexFlatL2(config.FAISS_DIMENSION)
        faiss_index = faiss.IndexIVFPQ(
            quantizer, config.FAISS_DIMENSION, config.FAISS_NLIST, config.FAISS_M, config.FAISS_NBITS
        )
        faiss_index.train(train_matrix)
    
    # langchain wrapper for faiss 
    vector_store = FAISS(
        embedding_function=embeddings,
        index=faiss_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # batching to prevent each insane amounts of API calls to Ollama
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in range(len(batch))]
        try:
            vector_store.add_documents(batch, ids=batch_ids)
            print(f"batch {i//batch_size + 1} ingested")
        except Exception as e:
            print(f"Error on batch {i//batch_size + 1}: {e}")

    vector_store.save_local(config.DB_PATH)
    print("Ingestion complete")

if __name__ == "__main__":
    build_and_save_index()