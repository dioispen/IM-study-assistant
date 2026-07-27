# Binary Search Tree

## Insertion

To insert a node into a binary search tree, compare the new key against the current node and recurse left when the new key is smaller or right when it is larger, until an empty spot is found for the new node.

## Deletion

Deleting a node with two children requires finding its in-order successor, the smallest node in its right subtree, and replacing the deleted node's key with the successor's key before removing the successor itself.
