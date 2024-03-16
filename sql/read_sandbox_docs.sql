SELECT
  e.document,
  e.cmetadata ->> 'title' title,
  e.cmetadata ->> 'day' "day"
FROM
  langchain_pg_embedding e
LEFT JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'sandbox'
ORDER BY "day" DESC
LIMIT %(limit)s