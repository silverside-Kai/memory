def retweet_prompt(link, llm, store_latest_raw):
    import pymongo
    from config import mongo_string
    from langchain.chains import RetrievalQAWithSourcesChain
    from modules.misc.max_length import truncate_string

    mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']

    query = {"link": link}
    relevant_doc = mongo_collection.find(query)

    retriever_latest_raw = store_latest_raw.as_retriever(search_kwargs={'k': 6})
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="map_reduce", retriever=retriever_latest_raw,
        return_source_documents=True)
    # this can probably be useful to improve QA: https://python.langchain.com/docs/use_cases/question_answering/#customizing-the-prompt

    tldr_with_tag = None
    for doc in relevant_doc:
        tldr_with_tag = doc['tldr_with_tag']

    bullet_points = [item["bullet_point"] for item in tldr_with_tag]

    query = f"""###What's the added value of this new content compared to the existing work?
    Start writing without replying to me. Just act as you directly start expressing your opinion about this new content in an appealing way.
    Write <250 chars. Try to use short sentences. Can you always start with pointing out the added value of the new content, instead of summarising it.
    New content: {bullet_points}"""

    try:
        llm_response = qa(query)
        result = llm_response["answer"]
        result = '🍄 Time for reverie... ' + result
        result = truncate_string(result, max_length=250)
        return result
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
