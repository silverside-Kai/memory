from langchain.vectorstores.pgvector import PGVector
from config import CONNECTION_STRING
from langchain.embeddings import HuggingFaceBgeEmbeddings


embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

store_basic_raw = PGVector(
    collection_name='basic_raw',
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

store_latest_raw = PGVector(
    collection_name='latest_raw',
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)