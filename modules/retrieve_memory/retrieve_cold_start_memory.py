def retrieve_cold_start_memory(llm, store_basic_raw, user_input, tag=None):
    from langchain.chains import RetrievalQA

    if tag:
        retriever_basic_raw = store_basic_raw.as_retriever(
            search_kwargs={'filter': {'tag': tag}})
    else:
        retriever_basic_raw = store_basic_raw.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever_basic_raw)

    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        result = llm_response["result"]
        return result
    except Exception as err:
        print('Exception occurred. Please try again', str(err))