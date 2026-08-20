# Hashing

## Hash Functions

A hash function maps a key onto a bucket index, and a good one spreads keys evenly across the table so that no single bucket collects far more keys than the rest.

## Amortized Cost

Lookup in a hash table costs constant time on average, and the resize that keeps the load factor low is amortized across the insertions that made the table grow.
