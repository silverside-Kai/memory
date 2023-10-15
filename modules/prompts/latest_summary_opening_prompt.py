def latest_summary_opening_prompt(num_contents):
    import openai
    from modules.misc.max_length import truncate_string
    import pymongo
    from config import mongo_string


    mongo_collection = pymongo.MongoClient(mongo_string)['mvp']['source']

    query = {"daily_tweeted": {"$exists": False}}
    sort_order = [("created_time", pymongo.DESCENDING)]

    data = mongo_collection.find(query).sort(sort_order).limit(num_contents)

    top_contents = ''
    for index, document in enumerate(data):
        top_content = f'{index+1}: ' + document['content'] + '\n' + document['content_long']
        top_content = truncate_string(top_content, max_length=500)
        top_contents = top_contents + top_content + '\n'

    prompt = f"""Summarise the above content in {num_contents} very short bullet points (each <10 words), as the prologue of AI Daily Pulse.
Use fewer than 270 characters in total.
"""
    prompt = top_contents + prompt

    prompts = [{"role": "user",
                "content": prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        request_timeout=30,
        messages=prompts
    )

    response_text = response['choices'][0]['message']['content']
    result = truncate_string(response_text, max_length=270)

    return result