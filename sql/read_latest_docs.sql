SELECT
  e.document,
  e.cmetadata ->> 'created_time' created_time
FROM
  langchain_pg_embedding e
LEFT JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'latest_raw'
ORDER BY created_time DESC
LIMIT %(limit)s