import config
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def chunk_markdown_document(markdown_text):
    # markdown attributes
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # splitting on headers and ensuring they are included for more context
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    
    heading_chunks = markdown_splitter.split_text(markdown_text)

    # further splits on paragraphs, sentences, and spaces for text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", r"(?<=\. )", " "],
        is_separator_regex=True
    )
    
    return text_splitter.split_documents(heading_chunks)