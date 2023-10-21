def insight_qa_prompt(link):
    import openai
    from config import openai_key, openai_api_base
    from modules.misc.max_length import truncate_string
    import os
    import pymongo
    from config import mongo_string


    os.environ['OPENAI_API_KEY'] = openai_key
    openai.api_key = openai_key
    openai.api_base = openai_api_base

    mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']

    query = {"link": link}
    data = mongo_collection.find(query)

    for document in data:
        top_content = document['content'] + '\n' + document['content_long']
        top_content = top_content[:500]
        top_link = document['link']

    prompt = """Generate a very short but interesting question about the AI content in 80 characters.
    And then answer this question in one simple sentence in 150 characters.
    Both Q and A should be fewer than 255 characters."""
    prompt = 'Content: ' + top_content + ' ' + prompt

    prompts = [{"role": "user",
                "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=30,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']
    print(response_text)
    result = truncate_string(response_text, max_length=255)
    result = result + ' ' + top_link

    return result