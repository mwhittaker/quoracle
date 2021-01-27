Quorums
=======

## Installation
TODO(mwhittaker): Make this package pip'able. For now, you have to clone and
install the dependencies yourself:

```
pip install -r requirements.txt
```

## Tutorial
Given a set of nodes `X`, a _read-write quorum system_ is a pair `(R, W)` where
`R` is a set of subsets of `X` called _read quorums_ and `W` is a set of
subsets of `X` called _write quorums_. A read-write quorum system satisfies the
property that every read quorum intersects every write quorum. This library
allows us to construct and analyze arbitrary read-write quorum systems. First,
we import the library.

```python
from quorums import *
```

Next, we specify the nodes in our quorum system. Our nodes can be strings,
integers, IP addresses, anything!

```python
a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')
```

Here, we construct a two by three grid of nodes. Every row is read quorum, and
one element from every row is a write quorum. Note that when we construct a
quorum system, we only have to specify the set of read quorums. The library
figures out the optimal set of write quorums automatically.

```python
grid = QuorumSystem(reads=a*b*c + d*e*f)
```

This prints `{'a', 'b', 'c'}` and `{'d', 'e', 'f'}`.

```python
for r in grid.read_quorums():
    print(r)
```

This prints `{'a', 'd'}`, `{'a', 'e'}`, `{'b', 'f'}`, `{'b', 'd'}`, ...

```python
for w in grid.write_quorums():
    print(w)
```

Alternatively, we could specify the write quorums...

```python
QuorumSystem(writes=(a + b + c) * (d + e + f))
```

or both the read and write quorums.

```python
QuorumSystem(reads=a*b*c + d*e*f, writes=(a + b + c) * (d + e + f))
```

We can check whether a given set is a read or write quorum. Note that any
superset of a quorum is also considered a quorum.

```python
grid.is_read_quorum({'a', 'b', 'c'})       # True
grid.is_read_quorum({'a', 'b', 'c', 'd'})  # True
grid.is_read_quorum({'a', 'b', 'd'})       # False

grid.is_write_quorum({'a', 'd'})      # True
grid.is_write_quorum({'a', 'd', 'd'}) # True
grid.is_write_quorum({'a', 'b'})      # False
```

The read resilience of our quorum system is the largest number `f` such that
despite the failure of any `f` nodes, we still have at least one read quorum.
Write resilience is defined similarly, and resilience is the minimum of read
and write resilience.

```python
grid.read_resilience()  # 1
grid.write_resilience() # 2
grid.resilience()       # 1
```

A _strategy_ is a discrete probability distribution over the set of read and
write quorums. A strategy gives us a way to pick quorums at random. The load of
a node is the probability that the node is selected by the strategy, and the
load of a strategy is the load of the most heavily loaded node. Using the
`strategy` method, we get a load-optimal strategy, i.e. the strategy with the
lowest possible load.

Typically in a distributed system, a read quorum of nodes is contacted to
perform a read, and a write quorum of nodes is contacted to perform a write.
Though we get to pick a strategy, we don't get to pick the fraction of
operations that are reads and the fraction of operations that are writes.  This
is determined by the workload. When constructing a strategy, we have to specify
the workload. The returned strategy is optimal only against this workload.
Here, we construct a strategy assuming that 75% of all operations are reads.

```python
strategy = grid.strategy(read_fraction=0.75)
```

We can use the strategy to sample read and write quorums.

```python
print(strategy.get_read_quorum())
print(strategy.get_read_quorum())
print(strategy.get_read_quorum())
print(strategy.get_write_quorum())
print(strategy.get_write_quorum())
print(strategy.get_write_quorum())
```

We can query the strategy's load.

```python
strategy.load(read_fraction=0.75) # 0.458
```

We can query the strategy's load on other workloads as well, though the
strategy may not be optimal.

```python
strategy.load(read_fraction=0)   # 0.333
strategy.load(read_fraction=0.5) # 0.416
strategy.load(read_fraction=1)   # 0.5
```

This is a shorthand for
`grid.strategy(read_fraction=0.25).load(read_fraction=0.25)`.

```python
grid.load(read_fraction=0.25) # 0.375
```

In the real world, we don't often have a fixed workload. Workloads change
over time. Instead of specifying a fixed read fraction, we can provide a
discrete probability distribution of read fractions. Here, we say that the
read fraction is 10% half the time and 75% half the time. `strategy` will
return the strategy that minimizes the expected load according to this
distribution.

```python
distribution = {0.1: 0.5, 0.75: 0.5}
strategy = grid.strategy(read_fraction=distribution)
strategy.load(read_fraction=distribution) # 0.404
```

We can also specify the write fraction instead of the read fraction, if we
prefer.

```python
strategy = grid.strategy(write_fraction=0.75)
strategy.load(write_fraction=distribution) # 0.429
```

In the real world, not all nodes are equal. We often run distributed systems on
heterogenous hardware, so some nodes might be faster than others. To model
this, we instatiate every node with its capacity. Here, nodes a, c, and e can
process 1000 commands per second, while nodes b, d, and f can only process 500
requests per second.

```python
a = Node('a', capacity=1000)
b = Node('b', capacity=500)
c = Node('c', capacity=1000)
d = Node('d', capacity=500)
e = Node('e', capacity=1000)
f = Node('f', capacity=500)
```

Now, load can be interpreted as the inverse of the peak throughput of the
quorum system. We can also call `capacity` to get this inverse directly.
Here, our quorum system is capable of processing 1333 commands per second for
a workload of 75% reads.

```python
grid = QuorumSystem(reads=a*b*c + d*e*f)
strategy = grid.strategy(read_fraction=0.75)
strategy.load(read_fraction=0.75)     # 0.00075
strategy.capacity(read_fraction=0.75) # 1333
```

Nodes might also process reads and writes at different speeds. We can specify
the peak read and write throughput of every node separately. Here, we assume
reads are ten times as fast as writes.

```python
a = Node('a', write_capacity=1000, read_capacity=10000)
b = Node('b', write_capacity=500, read_capacity=5000)
c = Node('c', write_capacity=1000, read_capacity=10000)
d = Node('d', write_capacity=500, read_capacity=5000)
e = Node('e', write_capacity=1000, read_capacity=10000)
f = Node('f', write_capacity=500, read_capacity=5000)
```

With 100% reads, our quorum system can process 10,000 commands per second.
This throughput decreases as we increase the fraction of writes.

```python
grid = QuorumSystem(reads=a*b*c + d*e*f)
grid.capacity(read_fraction=1)   # 10,000
grid.capacity(read_fraction=0.5) # 3913
grid.capacity(read_fraction=0)   # 2000
```

Another real world complication is the fact that machines sometimes fail and
are sometimes slow. If we contact a quorum of nodes, some of them may fail, and
we'll get stuck waiting to hear back from them. Or, some of them may be
stragglers, and we'll wait longer than we'd like. We can address this problem
by contacting more than the bare minimum number of nodes.

Formally, we say a read quorum (or write quorum) q is _f-resilient_ if despite
the failure of any f nodes, q still forms a read quorum (or write quorum). A
strategy is f-resilient if it only selects f-resilient quorums. By default,
`strategy` returns 0-resilient quorums. We can pass in the `f` argument to get
more resilient strategies.

```python
strategy = grid.strategy(read_fraction=0.5, f=1)
```

These sets are quorums even if 1 machine fails.

```python
strategy.get_read_quorum()
strategy.get_write_quorum()
```

Putting everything together, we can use this library to pick quorum systems
that are well suited to our workload. For example, say we're implementing a
distributed file system and want to pick a 5 node quorum system with a
resilience of 1 that has a good load on workloads that are 90% reads 90% of the
time and 10% reads 10% of the time. We can try out three quorum systems: a
simple majority quorum system, a crumbling walls quorum system, and a paths
quorum system.

```python
simple_majority = QuorumSystem(reads=majority([a, b, c, d, e]))
crumbling_walls = QuorumSystem(reads=a*b + c*d*e)
paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)
```

We make sure we have the desired resilience.

```python
assert(simple_majority.resilience() >= 1)
assert(crumbling_walls.resilience() >= 1)
assert(paths.resilience() >= 1)
```

We check the loads and see that the crumbling walls quorum system has the
highest load, so we use the crumbling walls quorum system to implement our file
system.

```python
distribution = {0.9: 0.9, 0.1: 0.1}
simple_majority.capacity(read_fraction=distribution) # 5089
crumbling_walls.capacity(read_fraction=distribution) # 5837
paths.capacity(read_fraction=distribution)           # 5725
```

Maybe some time later, we experiencing high latency because of stragglers and
want to switch to a 1-resilient strategy. We again compute the loads, but now
see that the simple majority quorum system has the highest load, so we switch
from the crumbling walls quorum system to the simple majority quorum system.

```python
simple_majority.capacity(read_fraction=distribution, f=1) # 3816
crumbling_walls.capacity(read_fraction=distribution, f=1) # 1908
paths.capacity(read_fraction=distribution, f=1)           # 1908
```
