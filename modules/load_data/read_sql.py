def read_sql(sql_file_name, num_docs=50):
    from config import pgvector_string
    from sqlalchemy import create_engine
    import pandas as pd

    postgresql_engine = create_engine(
        pgvector_string
    )

    with open('./sql/' + sql_file_name, "r") as sql_file:
        sql_code = sql_file.read()
    sql_code = sql_code.replace('\n', ' ')

    docs = pd.read_sql(sql_code, postgresql_engine, params={'limit': num_docs})

    return docs