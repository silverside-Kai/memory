/* global use, db */
// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.

// The current database to use.
use('mvp');

// Update the document based on the _id field
db.getCollection('source').updateOne(
    { "_id": ObjectId("64fb3ce8fa63d527d46276a0") }, // Match the document based on the _id field
    { $set: { "tweeted_timestamp": null } } // Add the new "tweeted_timestamp" field
  );