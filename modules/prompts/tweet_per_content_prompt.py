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
    # this can probably be useful to improve QA: https://python.langchain.com/docs/use_cases/question_answering/#customizing-the-prompt
    
    tldr_with_tag = None
    for doc in relevant_doc:
        tldr_with_tag = doc['tldr_with_tag']

    bullet_points = [item["bullet_point"] for item in tldr_with_tag]
    
    query = f"""###Prompt can you help organise the following bullet points
      into a tweet (NO HASHTAG) in very brief words (<250 chars) and quantitatively assess its importance with a score in a 0-10 range?
      Step 1: Put the importance score at the beginninng in the format "Importance Score: X/10". Replace X with your score.
      Please give the score following an aggressive (widespread) normal distribution with the mean as 6.
      An example of being very important is the release of GPT4. It's a 10. Because it's relevant to everyone.
      An example of being not important at all is any content trying to do marketing for companies. It's a 1. Because it's irrelevant for most people.
      Step 2: Summarise the content into bullet points each starting with one relevant emoji as bullet.
      If possible the first bullet point should be about why you think this content is important or not important. Given your reason.
      Bullet points: {bullet_points}"""

    try:
        llm_response = qa(query)
        result = llm_response["result"]
        while len(result) > 240:
            last_newline_index = result.rfind("\n")
            result = result[:last_newline_index]
        tweet = result + f""" #AI\n{link}"""
        return tweet
    except Exception as err:
        print('Exception occurred. Please try again', str(err))
    