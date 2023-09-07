/* global use, db */
// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.

// The current database to use.
use('mvp');

// Search for documents in the current collection.
db.getCollection('source').updateMany(
    {},
    {
      $unset: {
        tldr_with_tag: 1,
        vectordb_stored_timestamp_utc: 1
      }
    }
  );
