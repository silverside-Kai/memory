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
        daily_tweeted: 1,
        tweet_id: 1
      }
    }
  );
