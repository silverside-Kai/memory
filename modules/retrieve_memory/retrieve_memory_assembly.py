def retrieve_memory_assembly(llm, store_basic_raw, store_latest_raw, user_input, weight_latest, tag=None):
    from langchain.retrievers import EnsembleRetriever
    from langchain.chains import RetrievalQA

    if tag:
        retriever_basic_raw = store_basic_raw.as_retriever(
            search_kwargs={'filter': {'tag': tag}})
        retriever_latest_raw = store_latest_raw.as_retriever(
            search_kwargs={'filter': {'tag': tag}})
    else:
        retriever_basic_raw = store_basic_raw.as_retriever()
        retriever_latest_raw = store_latest_raw.as_retriever()
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_basic_raw, retriever_latest_raw],
                                           weights=[(1-weight_latest), weight_latest])
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=ensemble_retriever)

    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        result = llm_response["result"]
        return result
    except Exception as err:
        print('Exception occurred. Please try again', str(err))