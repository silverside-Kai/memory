def load_docs_excl_daily_tweeted(past_n_days):
    import pymongo
    from config import mongo_string
    from modules.misc.beginning_of_certain_date import beginning_of_certain_date

    mongo_db_name = 'mvp'
    mongo_collection_name = 'source'
    collection = pymongo.MongoClient(
        mongo_string)[mongo_db_name][mongo_collection_name]
    # Define your query (for example, find documents with a specific field value)
    query_recent_nonreflected = {
        "$and": [
            {"created_time": {"$gte": beginning_of_certain_date(
                past_n_days).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}},
            {"daily_tweeted": {"$exists": False}}
        ]
    }
    relevant_docs = collection.find(query_recent_nonreflected)

    return relevant_docs
