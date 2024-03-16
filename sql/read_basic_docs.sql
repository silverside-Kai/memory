SELECT
  e.document,
  e.cmetadata ->> 'source' source
FROM
  langchain_pg_embedding e
LEFT JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'basic_raw'
