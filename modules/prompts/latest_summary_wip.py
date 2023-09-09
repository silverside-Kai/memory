def latest_summary(llm, store_latest_raw, tag=None):
    from langchain.chains import RetrievalQA


    # maybe helpful: https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa#custom-prompts
    if tag:
        retriever_latest_raw = store_latest_raw.as_retriever(
            search_kwargs={'filter': {'tag': tag}})
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever_latest_raw)
        user_input = f"""
    Can you give a quick overview of what happened in the AI world especially on the topic "{tag}" after 2023-08-31 with the web link?
    Only select the content when you're sure it's after 2023-08-31.
    Compile the content with bullet points.
    """
    else:
        retriever_latest_raw = store_latest_raw.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever_latest_raw)
        user_input = f"""
    Can you give a quick overview of what happened in the AI world after 2023-08-31 with the web link?
    Only select the content when you're sure it's after 2023-08-31.
    Compile the content with bullet points.
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