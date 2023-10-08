def tag_meta():
    from config import pgvector_string
    from sqlalchemy import create_engine
    import pandas as pd


    pg_engine = create_engine(pgvector_string)
    query_string = """
    SELECT
        e.document,
        e.cmetadata
    FROM langchain_pg_embedding e
    LEFT JOIN langchain_pg_collection c
    ON e.collection_id = c.uuid WHERE c.name = 'tags'
    """
    tag_metadata = pd.read_sql(query_string, pg_engine)
    return(tag_metadata)