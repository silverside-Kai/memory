from config import mongo_string
from langchain.schema.document import Document
from modules.load_data.load_to_be_embedded_docs import load_to_be_embedded_docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.load_data.load_vector_stores import store_latest_raw
import pymongo
from bson import ObjectId
from datetime import datetime


class PipelineMemory:
    def __init__(self, mongo_db_name, mongo_collection_name, n_days):
        self.mongo_collection = pymongo.MongoClient(
            mongo_string)[mongo_db_name][mongo_collection_name]
        self.n_days = n_days

    # Step 3: store tldr into pgvector
    def write_to_pgvector(self):
        to_be_embedded_docs = load_to_be_embedded_docs(
            self.mongo_collection, self.n_days)

        for document in to_be_embedded_docs:
            filter = {"_id": ObjectId(document['_id'])}
            if 'text' in document:
                content = document['text']
                doc = Document(page_content=content,
                               metadata={"source": document["source"],
                                         "created_time": document["created_time"],
                                         "link": document["link"],
                                         "title": document["title"],
                                         "media": document["media"],
                                         "doc_id": str(document['_id'])})
                store_latest_raw.add_documents([doc])
            else:
                content = document['content']
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
                docs = [Document(page_content=x,
                                 metadata = {"source": document["source"],
                                             "created_time": document["created_time"],
                                             "link": document["link"],
                                             "title": document["title"],
                                             "media": document["media"],
                                             "doc_id": str(document['_id'])}) for x in text_splitter.split_text(content)]
                store_latest_raw.add_documents(docs)
            update_query = {
                "$set": {'vectordb_stored_timestamp_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')}
            }
            self.mongo_collection.update_one(filter, update_query)
            print(document['title'] + ' embedding done.')


if __name__ == '__main__':
    pipeline_memory = PipelineMemory(mongo_db_name='mvp',
                                     mongo_collection_name='source',
                                     n_days=3)
    pipeline_memory.write_to_pgvector()
