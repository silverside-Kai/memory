def load_relevant_contents(collection, n_days):
    from modules.beginning_of_certain_date import beginning_of_certain_date


    # Define your query (for example, find documents with a specific field value)
    query_recent_nonreflected = {
        "$and": [
            {"created_time": {"$gte": beginning_of_certain_date(
                n_days).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}},
            {"tldr_with_tag": {"$exists": False}}
        ]
    }
    relevant_docs = collection.find(query_recent_nonreflected)

    return relevant_docs
