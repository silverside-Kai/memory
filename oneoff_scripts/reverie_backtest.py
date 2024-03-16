from modules.load_data.read_sql import read_sql
from modules.load_data.load_vector_stores import embeddings, embeddings_base
from modules.prompts.simple_prompt import simple_prompt
import random
import pymongo
from config import mongo_string, pgvector_string, openai_key, openai_api_base
import pandas as pd
from langchain.vectorstores.pgvector import PGVector
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import openai
from sqlalchemy import create_engine
from datetime import datetime
from langchain.document_loaders import ArxivLoader


postgresql_engine = create_engine(
    pgvector_string
)

os.environ['OPENAI_API_KEY'] = openai_key
openai.api_key = openai_key
openai.api_base = openai_api_base

top_per_day = 1
regular_per_day = 5
days = 10
num_questions = 3
num_docs_per_retrieval_activator = 4
num_docs_per_retrieval_reverie = 15
formatted_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prompt_activate_memory = f'Break down the above information into {num_questions} AI problems. One problem per line. No empty line.'
prompt_knowledge_reverie = 'From the above content, what do you think are the most relevant techniques and solutions? Just explain in one paragaph.'

# Step 1: Simulate top / regular papers coming in batches
# top papers
data_top = read_sql('read_basic_docs.sql')
distinct_sources = data_top['source'].unique()
sampled_sources = random.sample(list(distinct_sources), top_per_day * days)
top_sample = data_top[data_top['source'].isin(sampled_sources)]
top_sample.rename(columns={'source': 'title'}, inplace=True)
top_sample = top_sample.groupby('title')['document'].agg(
    lambda x: ' '.join(x)).reset_index()
length_min_top = min(top_sample['document'].str.len())
# regular papers
mongo_collection = pymongo.MongoClient(
    mongo_string)['mvp']['source']
query = {
    "$and": [
        {"media": "paper"},
        {"$where": "this.content_long.length > 1000"}
    ]
}
data_regular = mongo_collection.find(query).limit(regular_per_day * days)
regular_sample_rows = []
for doc in data_regular:
    title = doc['title']
    link = doc['link']
    last_part = link.split("/")[-1]
    print(last_part)
    docs = ArxivLoader(query=last_part, load_max_docs=1, doc_content_chars_max=15000).load()
    document = docs[0].page_content
    regular_sample_rows.append({'document': document, 'title': title})
regular_sample = pd.DataFrame(regular_sample_rows)
length_min_regular = min(regular_sample['document'].str.len())
length_min = min(length_min_top, length_min_regular)

top_sample['document'] = top_sample['document'].str.slice(0, length_min)
regular_sample['document'] = regular_sample['document'].str.slice(
    0, length_min)

regular_sample.to_sql(name='regular_paper_samples', schema='sandbox', if_exists='replace', index=False, con=postgresql_engine, method='multi')

sampled_sources_copy = sampled_sources.copy()
top_sample_copy = top_sample.copy()
regular_sample_copy = regular_sample.copy()

sampled_sources = sampled_sources_copy.copy()
top_sample = top_sample_copy.copy()
regular_sample = regular_sample_copy.copy()
# Step 2: Simulate every day
random.seed(0)
results_rows = []
papers_rows = []
for day in range(days):
    print(f'Day {day} starts.')
    # Ingest top paper
    sampled_top = random.sample(sampled_sources, top_per_day)
    print(f'Top content: {sampled_top}')
    sampled_sources = [x for x in sampled_sources if x not in sampled_top]
    sampled_top_content = top_sample[top_sample['title'].isin(sampled_top)]
    # Ingest regular papers
    sampled_regular_content = regular_sample.sample(n=regular_per_day)
    sampled_regular_content_titles = sampled_regular_content['title'].tolist()
    print(f'Regular content: {sampled_regular_content_titles}')
    regular_sample = regular_sample.drop(index=sampled_regular_content.index)
    # Store content into vectordb
    if day == 0:
        store_sandbox = PGVector(
            collection_name='sandbox',
            connection_string=pgvector_string,
            embedding_function=embeddings_base,
            pre_delete_collection=True
        )
    else:
        store_sandbox = PGVector(
            collection_name='sandbox',
            connection_string=pgvector_string,
            embedding_function=embeddings_base,
            pre_delete_collection=False
        )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=10)
    # Store top paper
    for index, row in sampled_top_content.iterrows():
        document = row['document']
        title = row['title']
        docs = [Document(page_content=x,
                         metadata={"title": title,
                                   "day": day}) for x in text_splitter.split_text(document)]
        store_sandbox.add_documents(docs)
    # Store regular papers
    for index, row in sampled_regular_content.iterrows():
        document = row['document']
        title = row['title']
        docs = [Document(page_content=x,
                         metadata={"title": title,
                                   "day": day}) for x in text_splitter.split_text(document)]
        store_sandbox.add_documents(docs)
    print('Vector db stored.')
    # Ask questions using latest randomly sampled docs
    data = read_sql('read_sandbox_docs.sql', num_docs=1000000)
    data = data[data['day'] == str(day)]

    grouped_data = data.groupby('title')
    min_sample_size = 5 #grouped_data.size().min()
    sampled_data = grouped_data.apply(
        lambda group: group.sample(min_sample_size, replace=False))
    sampled_data = sampled_data.reset_index(drop=True)

    docs = ''
    for index, row in sampled_data.iterrows():
        line = row['document']
        new_str = ', '.join(line.split(', ')[1:])
        docs = docs + new_str + ' '

    prompt = docs + prompt_activate_memory

    response_text = simple_prompt(prompt)

    questions = response_text.splitlines()
    questions = [s for s in questions if len(s) >= 10]
    if len(questions)>num_questions:
        questions = questions[:num_questions]

    df = pd.DataFrame({"day": day,
                       "memory_activators": questions,
                       "created_time": formatted_datetime})

    df.to_sql(name='memory_activators', schema='sandbox', if_exists='append', index=False,
              con=postgresql_engine, method='multi')
    print('Memory activators generated.')
    # Use questions to search content and update ranking score
    page_contents = ''
    for i in range(len(questions)):
        if len(questions)==1:
            continue
        else:
            relevant_docs = store_sandbox.similarity_search_with_relevance_scores(
                questions[i], num_docs_per_retrieval_activator)
            print(f'Memory activator: ' + questions[i])
            for j in range(len(relevant_docs)):
                page_content = relevant_docs[j][0].page_content
                page_contents = page_contents + '\n' + page_content
    iterative_summary = simple_prompt(page_contents + '\n' + prompt_knowledge_reverie)
    print('Knowledge reverie: ' + iterative_summary)

    df_summary = pd.DataFrame({"day": day,
                               "knowledge_reveries": [iterative_summary],
                               "created_time": formatted_datetime})
    df_summary.to_sql(name='knowledge_reveries', schema='sandbox', if_exists='append', index=False,
                      con=postgresql_engine, method='multi')
    iterative_relevant_docs = store_sandbox.similarity_search_with_relevance_scores(
    iterative_summary, num_docs_per_retrieval_reverie)
    for k in range(len(iterative_relevant_docs)):
        title = iterative_relevant_docs[k][0].metadata['title']
        from_day = iterative_relevant_docs[k][0].metadata['day']
        relevance_score = iterative_relevant_docs[k][1]
        results_rows.append({'current_day': day,
                            'from_day': from_day,
                            'title': title,
                            'relevance_score': relevance_score})
results = pd.DataFrame(results_rows)
results['experiment_time'] = formatted_datetime
results.to_sql(name='relevance_score', schema='sandbox', if_exists='append', index=False,
               con=postgresql_engine, method='multi')
# Update network structure

# Step 3: Monitor the ranking scores of top / regular papers
