def evaluate_importance(llm, store_basic_raw, store_latest_raw, bullet_point, tag):
    from langchain.retrievers import EnsembleRetriever
    from langchain.chains import RetrievalQA


    retriever_basic_raw = store_basic_raw.as_retriever(
        search_kwargs={'filter': {'tag': tag}})
    retriever_latest_raw = store_latest_raw.as_retriever(
        search_kwargs={'filter': {'tag': tag}})
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_basic_raw, retriever_latest_raw],
                                           weights=[0.5, 0.5])
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=ensemble_retriever)
    user_input = f"""
        What do you think is the importance of this content "{bullet_point}" in the domain of "{tag}".
        Please give a score from 0-10.
        And point out its added value to the topic "{tag}" and relate to the classic contents in the past.
        """
    print(user_input)

    # call the OpenAI API
    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        result = llm_response["result"]
        print(result)
        return result
    except Exception as err:
        print('Exception occurred. Please try again', str(err))