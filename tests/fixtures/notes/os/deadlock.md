# Deadlock

## Conditions

Deadlock arises when four conditions hold at once: mutual exclusion, where a resource can be held by only one process; hold and wait, where a process holds a resource while waiting for another; no preemption, where a resource cannot be forcibly taken from a process; and circular wait, where a cycle of processes each wait for a resource held by the next. Breaking any single one of these four conditions is enough to prevent deadlock from occurring in a running system.
