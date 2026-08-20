# Paging

## Page Table

Paging splits a process address space into fixed size pages, and a page table maps each virtual page onto the physical frame that currently holds it in memory.

## Page Faults

A page fault happens when a process touches a page that is not resident, and the operating system loads that page from disk before restarting the faulting instruction.
