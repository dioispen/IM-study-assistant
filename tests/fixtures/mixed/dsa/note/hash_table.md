# Hash Table

## Buckets

A hash table stores each key in a bucket chosen by a hash function, so a lookup goes straight to one bucket instead of scanning the whole table for the key.

## Collisions

Two keys can hash into the same bucket, which is a collision, and the table needs a rule for what happens next rather than losing one of the two keys.

## Chaining

Chaining resolves a collision by keeping a linked list in each bucket, so every key that hashes there is appended to that bucket list and found by walking it.

## Open Addressing

Open addressing resolves a collision by probing for another empty bucket in the table itself, so no bucket holds a list and every key sits in a bucket of its own.

## Load Factor

The load factor is the number of keys divided by the number of buckets, and a table is resized once it grows past a threshold because probe lengths climb sharply after that.
