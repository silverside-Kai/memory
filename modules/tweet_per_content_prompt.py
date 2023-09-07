def tweet_per_content_prompt(link, llm, store_basic_raw, store_latest_raw, weight_latest):
    import pymongo
    from config import mongo_string
    from langchain.retrievers import EnsembleRetriever
    from langchain.chains import RetrievalQA


    mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']

    query = {"link": link}
    relevant_doc = mongo_collection.find(query)

    retriever_basic_raw = store_basic_raw.as_retriever()
    retriever_latest_raw = store_latest_raw.as_retriever()
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_basic_raw, retriever_latest_raw],
                                           weights=[(1-weight_latest), weight_latest])
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=ensemble_retriever)
    
    tldr_with_tag = None
    for doc in relevant_doc:
        tldr_with_tag = doc['tldr_with_tag']
    
    query = f"""###Prompt can you help organise the following content
      into a tweet and quantitatively assess its importance in 
      very brief words with a score in a 0-10 range? '{tldr_with_tag}'"""

    try:
        llm_response = qa(query)
        result = llm_response["result"]
        tweet = result + f"""\n{link}"""
        return tweet
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
    