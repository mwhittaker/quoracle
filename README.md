Quoracle
========

Quoracle is a library for constructing and analyzing [read-write quorum
systems](https://scholar.google.com/scholar?cluster=4847365665094368145). Run
`pip install quoracle` and then follow along with the tutorial below to get
started.

## Quorum Systems
Given a set of nodes `X`, a _read-write quorum system_ is a pair `(R, W)` where

1. `R` is a set of subsets of `X` called _read quorums_,
2. `W` is a set of subsets of `X` called _write quorums_, and
3. every read quorum intersects every write quorum.

quoracle allows us to construct and analyze arbitrary read-write quorum
systems. First, we import the library.

```python
from quoracle import *
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

Now, we construct a two by three grid of nodes. Every row is read quorum, and
one element from every row is a write quorum. Note that when we construct a
quorum system, we only have to specify the set of read quorums. The library
figures out the optimal set of write quorums automatically.

```python
grid = QuorumSystem(reads=a*b*c + d*e*f)
```

This next code snippet prints out the read quorums `{'a', 'b', 'c'}` and `{'d',
'e', 'f'}`.

```python
for r in grid.read_quorums():
    print(r)
```

And this next code snippet prints out the write quorums `{'a', 'd'}`, `{'a',
'e'}`, `{'b', 'f'}`, `{'b', 'd'}`, ...

```python
for w in grid.write_quorums():
    print(w)
```

Alternatively, we can construct a quorum system be specifying the write
quorums.

```python
QuorumSystem(writes=(a + b + c) * (d + e + f))
```

Or, we can specify both the read and write quorums.

```python
QuorumSystem(reads=a*b*c + d*e*f, writes=(a + b + c) * (d + e + f))
```

But, remember that every read quorum must intersect every write quorum. If we
try to construct a quorum system with non-overlapping quorums, an exception
will be thrown.

```python
QuorumSystem(reads=a+b+c, writes=d+e+f)
# ValueError: Not all read quorums intersect all write quorums
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

## Resilience
The _read resilience_ of our quorum system is the largest number `f` such that
despite the failure of any `f` nodes, we still have at least one read quorum.
_Write resilience_ is defined similarly, and _resilience_ is the minimum of
read and write resilience.

Here, we print out the read resilience, write resilience, and resilience of our
grid quorum system. We can fail any one node and still have a read quorum, but
if we fail one node from each row, we eliminate every read quorum, so the read
resilience is 1. Similarly, we can fail any two nodes and still have a write
quorum, but if we fail one node from every column, we eliminate every write
quorum, so our write resilience is 1. The resilience is the minimum of 1 and 2,
which is 1.

```python
grid.read_resilience()  # 1
grid.write_resilience() # 2
grid.resilience()       # 1
```

## Strategies
A _strategy_ is a discrete probability distribution over the set of read and
write quorums. A strategy gives us a way to pick quorums at random. We'll see
how to construct optimal strategies in a second, but for now, we'll construct a
strategy by hand. To do so, we have to provide a probability distribution over
the read quorums and a probability distribution over the write quorums. Here,
we'll pick the top row twice as often as the bottom row, and we'll pick each
column uniformly at random. Note that when we specify a probability
distribution, we don't have to provide exact probabilities. We can simply pass
in weights, and the library will automatically normalize the weights into a
valid probability distribution.

```python
# The read quorum strategy.
sigma_r = {
    frozenset({'a', 'b', 'c'}): 2.,
    frozenset({'d', 'e', 'f'}): 1.,
}

# The write quorum strategy.
sigma_w = {
    frozenset({'a', 'd'}): 1.,
    frozenset({'b', 'e'}): 1.,
    frozenset({'c', 'f'}): 1.,
}
strategy = grid.make_strategy(sigma_r, sigma_w)
```

Once we have a strategy, we can use it to sample read and write quorums. Here,
we expect `get_read_quorum` to return the top row twice as often as the bottom
row, and we expect `get_write_quorum` to return every column uniformly at
random.

```python
print(strategy.get_read_quorum())
print(strategy.get_read_quorum())
print(strategy.get_read_quorum())
print(strategy.get_read_quorum())
print(strategy.get_write_quorum())
print(strategy.get_write_quorum())
print(strategy.get_write_quorum())
print(strategy.get_write_quorum())
```

## Load and Capacity
Typically in a distributed system, a read quorum of nodes is contacted to
perform a read, and a write quorum of nodes is contacted to perform a write.
Assume we have a workload with a _read fraction_ `fr` of reads and a _write
fraction_ `fw = 1 - fr` of writes. Given a strategy, the _load of a node_ is
the probability that the node is selected by the strategy. The _load of a
strategy_ is the load of the most heavily loaded node. The _load of a quorum
system_ is the load of the optimal strategy, i.e. the strategy that achieves
the lowest load. The most heavily loaded node in a quorum system is a
throughput bottleneck, so the lower the load the better.

Let's calculate the load of our strategy assuming a 100% read workload (i.e. a
workload with a read fraction of 1).

- The load of `a` is 2/3 because the read quorum `{a, b, c}` is chosen 2/3 of
  the time.
- The load of `b` is 2/3 because the read quorum `{a, b, c}` is chosen 2/3 of
  the time.
- The load of `c` is 2/3 because the read quorum `{a, b, c}` is chosen 2/3 of
  the time.
- The load of `d` is 1/3 because the read quorum `{d, e, f}` is chosen 2/3 of
  the time.
- The load of `e` is 1/3 because the read quorum `{d, e, f}` is chosen 2/3 of
  the time.
- The load of `f` is 1/3 because the read quorum `{d, e, f}` is chosen 2/3 of
  the time.

The largest node load is 2/3, so our strategy has a load of 2/3. Rather than
calculating load by hand, we can simply call the `load` function.

```python
print(strategy.load(read_fraction=1)) # 2/3
```

Now let's calculate the load of our strategy assuming a 100% write workload.
Again, we calculate the load on every node.

- The load of `a` is 1/3 because the write quorum `{a, d}` is chosen 1/3 of
  the time.
- The load of `b` is 1/3 because the write quorum `{b, e}` is chosen 1/3 of
  the time.
- The load of `c` is 1/3 because the write quorum `{c, f}` is chosen 1/3 of
  the time.
- The load of `d` is 1/3 because the write quorum `{a, d}` is chosen 1/3 of
  the time.
- The load of `e` is 1/3 because the write quorum `{b, e}` is chosen 1/3 of
  the time.
- The load of `f` is 1/3 because the write quorum `{c, f}` is chosen 1/3 of
  the time.

The largest node load is 1/3, so our strategy has a load of 1/3. Again, rather
than calculating load by hand, we can simply call the `load` function. Note
that we can pass in a `read_fraction` or `write_fraction` but not both.

```python
print(strategy.load(write_fraction=1)) # 1/3
```

Now let's calculate the load of our strategy on a 25% read and 75% write
workload.

- The load of `a` is `0.25 * 2/3 + 0.75 * 1/3 = 5/12` because 25% of the time
  we perform a read and select the read quorum `{a, b, c}` with 2/3 probability
  and 75% of the time, we perform a write and select the write quorum `{a, d}`
  with 1/3 probability.
- The load of `b` is `0.25 * 2/3 + 0.75 * 1/3 = 5/12` because 25% of the time
  we perform a read and select the read quorum `{a, b, c}` with 2/3 probability
  and 75% of the time, we perform a write and select the write quorum `{b, e}`
  with 1/3 probability.
- The load of `c` is `0.25 * 2/3 + 0.75 * 1/3 = 5/12` because 25% of the time
  we perform a read and select the read quorum `{a, b, c}` with 2/3 probability
  and 75% of the time, we perform a write and select the write quorum `{c, f}`
  with 1/3 probability.
- The load of `d` is `0.25 * 1/3 + 0.75 * 1/3 = 1/3` because 25% of the time
  we perform a read and select the read quorum `{d, e, f}` with 2/3 probability
  and 75% of the time, we perform a write and select the write quorum `{a, d}`
  with 1/3 probability.
- The load of `e` is `0.25 * 1/3 + 0.75 * 1/3 = 1/3` because 25% of the time
  we perform a read and select the read quorum `{d, e, f}` with 2/3 probability
  and 75% of the time, we perform a write and select the write quorum `{b, e}`
  with 1/3 probability.
- The load of `f` is `0.25 * 1/3 + 0.75 * 1/3 = 1/3` because 25% of the time
  we perform a read and select the read quorum `{d, e, f}` with 2/3 probability
  and 75% of the time, we perform a write and select the write quorum `{c, f}`
  with 1/3 probability.

The largest node load is 5/12, so our strategy has a load of 5/12. At this
point, you can see that calculating load by hand is extremely tedious. We could
have skipped all that work and called `load` instead!

```python
print(strategy.load(read_fraction=0.25)) # 5/12
```

We can also compute the load on every node.

```python
print(strategy.node_load(a, read_fraction=0.25)) # 5/12
print(strategy.node_load(b, read_fraction=0.25)) # 5/12
print(strategy.node_load(c, read_fraction=0.25)) # 5/12
print(strategy.node_load(d, read_fraction=0.25)) # 1/3
print(strategy.node_load(e, read_fraction=0.25)) # 1/3
print(strategy.node_load(f, read_fraction=0.25)) # 1/3
```

Our strategy has a load of 5/12 on a 25% read workload, but what about the
quorum system? The quorum system does __not__ have a load of 5/12 because our
strategy is not optimal. We can call the `strategy` function to compute the
optimal strategy automatically.

```python
strategy = grid.strategy(read_fraction=0.25)
print(strategy)
# Strategy(reads={('a', 'b', 'c'): 0.5,
#                 ('d', 'e', 'f'): 0.5},
#          writes={('a', 'f'): 0.33333333,
#                  ('b', 'e'): 0.33333333,
#                  ('c', 'd'): 0.33333333})
print(strategy.load(read_fraction=0.25)) # 3/8
```

Here, we see that the optimal strategy picks all rows and all columns
uniformly. This strategy has a load of 3/8 on the 25% read workload. Since this
strategy is optimal, that means our quorum system also has a load of 3/8 on a
25% workload.

We can also query this strategy's load on other workloads as well. Note that
this strategy is optimal for a read fraction of 25%, but it may not be optimal
for other read fractions.

```python
print(strategy.load(read_fraction=0))   # 1/3
print(strategy.load(read_fraction=0.5)) # 5/12
print(strategy.load(read_fraction=1))   # 1/2
```

We can also use a quorum system's `load` function. The code snippet below is a
shorthand for `grid.strategy(read_fraction=0.25).load(read_fraction=0.25)`.

```python
grid.load(read_fraction=0.25) # 0.375
```

The capacity of strategy or quorum is simply the inverse of the load. Our
quorum system has a load of 3/8 on a 25% read workload, so it has a capacity of
8/3.

```python
print(grid.capacity(read_fraction=0.25)) # 8/3
```

The _capacity_ of a quorum system is proportional to the maximum throughput
that it can achieve before a node becomes bottlenecked. Here, if every node
could process 100 commands per second, then our quorum system could process
800/3 commands per second.

## Workload Distributions
In the real world, we don't often have a workload with a fixed read fraction.
Workloads change over time. Instead of specifying a fixed read fraction, we can
provide a discrete probability distribution of read fractions. Here, we say
that the read fraction is 10% half the time and 75% half the time. `strategy`
will return the strategy that minimizes the expected load according to this
distribution.

```python
distribution = {0.1: 1, 0.75: 1}
strategy = grid.strategy(read_fraction=distribution)
strategy.load(read_fraction=distribution) # 0.404
```

## Heterogeneous Node
In the real world, not all nodes are equal. We often run distributed systems on
heterogeneous hardware, so some nodes might be faster than others. To model
this, we instantiate every node with its capacity. Here, nodes `a`, `c`, and
`e` can process 1000 commands per second, while nodes `b`, `d`, and `f` can
only process 500 requests per second.

```python
a = Node('a', capacity=1000)
b = Node('b', capacity=500)
c = Node('c', capacity=1000)
d = Node('d', capacity=500)
e = Node('e', capacity=1000)
f = Node('f', capacity=500)
```

Now, the definition of capacity becomes much simpler. The capacity of a quorum
system is simply the maximum throughput that it can achieve. The load can be
interpreted as the inverse of the capacity. Here, our quorum system is capable
of processing 1333 commands per second for a workload of 75% reads.

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

# `f`-resilient Strategies
Another real world complication is the fact that machines sometimes fail and
are sometimes slow. If we contact a quorum of nodes, some of them may fail, and
we'll get stuck waiting to hear back from them. Or, some of them may be
stragglers, and we'll wait longer than we'd like. We can address this problem
by contacting more than the bare minimum number of nodes.

Formally, we say a read quorum (or write quorum) q is _`f`-resilient_ if
despite the failure of any `f` nodes, q still forms a read quorum (or write
quorum). A strategy is `f`-resilient if it only selects `f`-resilient quorums.
By default, `strategy` returns `0`-resilient quorums. We can pass in the `f`
argument to get more resilient strategies.

```python
strategy = grid.strategy(read_fraction=0.5, f=1)
```

These sets are quorums even if 1 machine fails.

```python
strategy.get_read_quorum()
strategy.get_write_quorum()
```

## Latency
TODO(mwhittaker): Write.

## Network Load
TODO(mwhittaker): Write.

## Search
TODO(mwhittaker): Write.

## Case Study
TODO(mwhittaker): Update.

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
