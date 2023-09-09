import os
from config import openai_key, openai_api_base, mongo_string, CONNECTION_STRING
import openai
from langchain.schema.document import Document
from modules.load_data.load_relevant_contents import load_relevant_contents
from modules.load_data.load_to_be_embedded_docs import load_to_be_embedded_docs
from modules.prompts.tldr import tldr
from modules.prompts.tagify import tagify
from modules.load_data.tag_meta import tag_meta
from modules.load_data.load_vector_stores import store_latest_raw
import pymongo
from bson import ObjectId
from datetime import datetime


class PipelineMemory:
    def __init__(self, mongo_db_name, mongo_collection_name, n_days, max_content_len):
        self.mongo_collection = pymongo.MongoClient(
            mongo_string)[mongo_db_name][mongo_collection_name]
        self.n_days = n_days
        self.max_content_len = max_content_len

    # Step 1: load all the (or only relevant) documents
    def load_contents(self):
        self.relevant_docs = load_relevant_contents(
            collection=self.mongo_collection, n_days=self.n_days)

    # Step 2.1: prompt to get tldr bullet points about the loaded documents
    # Step 2.2: prompt to categorise each bullet point per content
    def loop_over_contents(self):
        for document in self.relevant_docs:
            filter = {"_id": ObjectId(document['_id'])}

            prompt_tldr = tldr(document['media'],
                               document['title'],
                               document['content_long'][:self.max_content_len])

            response_tldr = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                request_timeout=30,
                messages=prompt_tldr
            )

            # divide response to seperate bullet points
            response_text = response_tldr['choices'][0]['message']['content']
            bullet_points = response_text.split('\n')
            bullet_points = [point.lstrip('- ').strip()
                             for point in bullet_points if point.strip()]

            tag_metadata = tag_meta()

            tldr_with_tag_objects = []
            for bullet_point in bullet_points:
                prompt_tag = tagify(bullet_point, tag_metadata)
                response_tag = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    request_timeout=30,
                    messages=prompt_tag
                )
                tag = response_tag['choices'][0]['message']['content']
                tldr_with_tag_object = {
                    'bullet_point': bullet_point,
                    'tag': tag
                }
                tldr_with_tag_objects.append(tldr_with_tag_object)

            # Update the document to add the new field
            update_query = {"$set": {'tldr_with_tag': tldr_with_tag_objects}}
            self.mongo_collection.update_one(filter, update_query)
            print(document['title'] + ' tldr_with_tag done.')

    # Step 3: store tldr into pgvector
    def write_to_pgvector(self):
        to_be_embedded_docs = load_to_be_embedded_docs(
            self.mongo_collection, self.n_days)

        for document in to_be_embedded_docs:
            filter = {"_id": ObjectId(document['_id'])}
            contents = document['tldr_with_tag']
            for ct in contents:
                content = ct['bullet_point']
                content = f'By {document["source"]} at {document["created_time"]} with the link {document["link"]}, {content}'
                tag = ct['tag']
                doc = Document(page_content=content,
                               metadata={"source": document["source"],
                                         "created_time": document["created_time"],
                                         "tag": tag,
                                         "link": document['link'],
                                         "title": document['title']})
                store_latest_raw.add_documents([doc])
            update_query = {
                "$set": {'vectordb_stored_timestamp_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')}
            }
            self.mongo_collection.update_one(filter, update_query)
            print(document['title'] + ' embedding done.')


if __name__ == '__main__':

    os.environ['OPENAI_API_KEY'] = openai_key
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = openai_api_base

    pipeline_memory = PipelineMemory(mongo_db_name='mvp',
                                     mongo_collection_name='source',
                                     n_days=1,
                                     max_content_len=10000)
    pipeline_memory.load_contents()
    pipeline_memory.loop_over_contents()
    pipeline_memory.write_to_pgvector()
