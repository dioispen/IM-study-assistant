# Virtual Memory

## Address Translation

Address translation turns a virtual address into a physical one by splitting it into a page number and an offset, then looking the page number up inside the page table.

## Translation Lookaside Buffer

A translation lookaside buffer caches recent page table entries so that a translation usually costs no extra memory reference, which is what keeps paging affordable.
