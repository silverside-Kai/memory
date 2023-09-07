def load_to_be_embedded_docs(collection, n_days):
    from modules.beginning_of_certain_date import beginning_of_certain_date


    query_reflected_nonstored = {
        "$and": [
            {"created_time": {"$gte": beginning_of_certain_date(
                n_days).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}},
            {"tldr_with_tag": {"$exists": True}},
            {"vectordb_stored_timestamp_utc": {"$exists": False}}
        ]
    }
    result_vector_storage = collection.find(query_reflected_nonstored)

    return result_vector_storage
