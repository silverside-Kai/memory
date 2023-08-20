import os
from config import openai_key, mongo_string, CONNECTION_STRING
import openai
from datetime import datetime, timedelta

## Loading Environment Variables
from typing import List, Tuple
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.schema.document import Document

import pymongo
from bson import ObjectId

os.environ['OPENAI_API_KEY'] = openai_key
openai.api_key = openai_key

# Step 1: load all the (or only relevant) documents
client = pymongo.MongoClient(mongo_string)
# Access a specific database
db = client["mvp"]
# Access a specific collection within the database
collection = db["source"]
# Define your query (for example, find documents with a specific field value)
def beginning_of_certain_date(n):
    certain_date = datetime.now() - timedelta(days=n)
    beginning = datetime(certain_date.year, certain_date.month, certain_date.day, 0, 0, 0)
    return beginning

query_recent_nonreflected = {
    "$and": [
        {"created_time": {"$gte": beginning_of_certain_date(20).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}},
        {"reflection": {"$exists": False}}
    ]
}

# Step 2: prompt to get high-level reflections about the loaded documents
# Process the result
def reflection(media, title, content):
    # call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": f"What do you think is the contribution of this {media}?\
                 Can you explain me like I'm five years old in one short sentence?\
                 You can directly start explaining without responding me.\
                 title: {title}. abstract: {content}"}
            ]
        )
    return response

# Execute the query and retrieve the result
result_reflection = collection.find(query_recent_nonreflected)
for document in result_reflection:
    filter = {"_id": ObjectId(document['_id'])}

    # Define the new field and its value
    new_field = "reflection"
    new_value = reflection(document['media'],
                           document['title'],
                           document['content'])

    # Update the document to add the new field
    update_query = {"$set": {new_field: new_value}}
    collection.update_one(filter, update_query)
    print(new_value)

# Step 3: store reflections into pgvector
query_reflected_nonstored = {
    "$and": [
        {"reflection": {"$exists": True}},
        {"stored": {"$exists": False}}
    ]
}

result_vector_storage = collection.find(query_reflected_nonstored)

embeddings = OpenAIEmbeddings()
collection_name = 'reflections'
store = PGVector(
    collection_name=collection_name,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
for document in result_vector_storage:
    filter = {"_id": ObjectId(document['_id'])}

    reflection_content = document['reflection']['choices'][0]['message']['content']
    doc = Document(page_content=reflection_content, metadata={"source": "local"})
    store.add_documents([doc])
    new_field = "stored"
    new_value = collection_name
    update_query = {"$set": {new_field, new_value}}

    collection.update_one(filter, update_query)