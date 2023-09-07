import os
from config import CONNECTION_STRING
from langchain.vectorstores.pgvector import PGVector
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Step 1: Generate tag metadata and store in the vectordb
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

store_tags = PGVector(
    collection_name='tags',
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

tag_metadata = [
    {
        "tag": "The foundational LLM",
        "explanation": """
 - Open-Sourced LLMs: These are LLMs whose architecture and often their pre-trained weights are publicly available.
 - Non-Open-Sourced LLMs: These LLMs are proprietary and often developed by corporations or research institutions.
"""
    },
    {
        "tag": "LLM improvements",
        "explanation": """
 - Fine-Tuning: The process of adapting a pre-trained LLM to a specific task or dataset by continuing the training process on specialized data.
Agent Related:
"""
    },
    {
        "tag": "Agent related",
        "explanation": """
Research and development focused on using LLMs in interactive agents such as chatbots, personal assistants, or other types of agents that interact with environments, users or other agents
 - Task Solving: Developing methods or framework for applying LLMs to solve specific tasks
 - Retrieval: Methods for effective data or information retrieval using LLMs
 - Reasoning: Algorithms and techniques to improve the logical reasoning capabilities of LLMs
 - Decision Making: Research focused on enabling LLMs to make decisions, possibly in a multi-agent setup or in an environment with dynamic variables
 - Reflection: synthesizes memories into higher level inferences over time and guides the agent’s future behavior
"""
    },
    {
        "tag": "Other data modalities",
        "explanation": """
 - Image Generation Models
 - Video Generation Models
 - Audio Generation Models
"""
    }
]

for tg in tag_metadata:
    doc = Document(page_content=tg["explanation"],
                   metadata={"tag": tg["tag"]})
    store_tags.add_documents([doc])

# Step 2: Generate doc folders based on tags
tag_list = [entry['tag'] for entry in tag_metadata]
base_path = './docs'  # Replace with your desired directory path
for tag in tag_list:
    folder_name = tag.lower().replace(' ', '_')
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        # Create the folder and its parents if they don't exist
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

# Step 3: Manually add documents in each folder

# Step 4: Load all the documents to vectordb
for tag in tag_list:
    folder_name = tag.lower().replace(' ', '_')
    folder_path = os.path.join(base_path, folder_name)

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            loader = PyMuPDFLoader(os.path.join(folder_path, filename))
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=10)
            texts = text_splitter.split_documents(documents)

            for document in texts:
                if hasattr(document, 'metadata'):
                    document.metadata['tag'] = tag
                else:
                    document.metadata = {'tag': tag}

            vectordb = PGVector.from_documents(documents=texts,
                                               embedding=embeddings,
                                               collection_name='basic_raw',
                                               connection_string=CONNECTION_STRING)
