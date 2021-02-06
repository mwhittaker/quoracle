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
>>> a = Node('a')
>>> b = Node('b')
>>> c = Node('c')
>>> d = Node('d')
>>> e = Node('e')
>>> f = Node('f')
```

Now, we construct a two by three grid of nodes. Every row is read quorum, and
one element from every row is a write quorum. Note that when we construct a
quorum system, we only have to specify the set of read quorums. The library
figures out the optimal set of write quorums automatically.

```python
>>> grid = QuorumSystem(reads=a*b*c + d*e*f)
```

This next code snippet prints out the read quorums.

```python
>>> for r in grid.read_quorums():
...     print(r)
{'a', 'b', 'c'}
{'d', 'e', 'f'}
```

And this next code snippet prints out the write quorums.

```python
>>> for w in grid.write_quorums():
...     print(w)
{'a', 'd'}
{'a', 'e'}
{'a', 'f'}
{'b', 'd'}
{'b', 'e'}
{'b', 'f'}
{'c', 'd'}
{'c', 'e'}
{'c', 'f'}
```

Alternatively, we can construct a quorum system be specifying the write
quorums.

```python
>>> QuorumSystem(writes=(a + b + c) * (d + e + f))
```

Or, we can specify both the read and write quorums.

```python
>>> QuorumSystem(reads=a*b*c + d*e*f, writes=(a + b + c) * (d + e + f))
```

But, remember that every read quorum must intersect every write quorum. If we
try to construct a quorum system with non-overlapping quorums, an exception
will be thrown.

```python
>>> QuorumSystem(reads=a+b+c, writes=d+e+f)
Traceback (most recent call last):
...
ValueError: Not all read quorums intersect all write quorums
```

We can check whether a given set is a read or write quorum. Note that any
superset of a quorum is also considered a quorum.

```python
>>> grid.is_read_quorum({'a', 'b', 'c'})
True
>>> grid.is_read_quorum({'a', 'b', 'c', 'd'})
True
>>> grid.is_read_quorum({'a', 'b', 'd'})
False
>>>
>>> grid.is_write_quorum({'a', 'd'})
True
>>> grid.is_write_quorum({'a', 'd', 'd'})
True
>>> grid.is_write_quorum({'a', 'b'})
False
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
>>> grid.read_resilience()
1
>>> grid.write_resilience()
2
>>> grid.resilience()
1
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
>>> # The read quorum strategy.
>>> sigma_r = {
...     frozenset({'a', 'b', 'c'}): 2.,
...     frozenset({'d', 'e', 'f'}): 1.,
... }
>>>
>>> # The write quorum strategy.
>>> sigma_w = {
...     frozenset({'a', 'd'}): 1.,
...     frozenset({'b', 'e'}): 1.,
...     frozenset({'c', 'f'}): 1.,
... }
>>> strategy = grid.make_strategy(sigma_r, sigma_w)
```

Once we have a strategy, we can use it to sample read and write quorums. Here,
we expect `get_read_quorum` to return the top row twice as often as the bottom
row, and we expect `get_write_quorum` to return every column uniformly at
random.

```python
>>> strategy.get_read_quorum()
{'a', 'b', 'c'}
>>> strategy.get_read_quorum()
{'a', 'b', 'c'}
>>> strategy.get_read_quorum()
{'d', 'e', 'f'}
>>> strategy.get_write_quorum()
{'b', 'e'}
>>> strategy.get_write_quorum()
{'c', 'f'}
>>> strategy.get_write_quorum()
{'b', 'e'}
>>> strategy.get_write_quorum()
{'a', 'd'}
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
>>> strategy.load(read_fraction=1)
0.6666666666666666
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
>>> strategy.load(write_fraction=1)
0.3333333333333333
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
>>> strategy.load(read_fraction=0.25)
0.41666666666666663
```

We can also compute the load on every node.

```python
>>> print(strategy.node_load(a, read_fraction=0.25))
0.41666666666666663
>>> print(strategy.node_load(b, read_fraction=0.25))
0.41666666666666663
>>> print(strategy.node_load(c, read_fraction=0.25))
0.41666666666666663
>>> print(strategy.node_load(d, read_fraction=0.25))
0.3333333333333333
>>> print(strategy.node_load(e, read_fraction=0.25))
0.3333333333333333
>>> print(strategy.node_load(f, read_fraction=0.25))
0.3333333333333333
```

Our strategy has a load of 5/12 on a 25% read workload, but what about the
quorum system? The quorum system does __not__ have a load of 5/12 because our
strategy is not optimal. We can call the `strategy` function to compute the
optimal strategy automatically.

```python
>>> strategy = grid.strategy(read_fraction=0.25)
>>> strategy
Strategy(reads={('a', 'b', 'c'): 0.5, ('d', 'e', 'f'): 0.5}, writes={('a', 'f'): 0.33333333, ('b', 'e'): 0.33333333, ('c', 'd'): 0.33333333})
>>> strategy.load(read_fraction=0.25))
0.3749999975
```

Here, we see that the optimal strategy picks all rows and all columns
uniformly. This strategy has a load of 3/8 on the 25% read workload. Since this
strategy is optimal, that means our quorum system also has a load of 3/8 on a
25% workload.

We can also query this strategy's load on other workloads as well. Note that
this strategy is optimal for a read fraction of 25%, but it may not be optimal
for other read fractions.

```python
>>> strategy.load(read_fraction=0)
0.33333333
>>> strategy.load(read_fraction=0.5)
0.416666665
>>> strategy.load(read_fraction=1)
0.5
```

We can also use a quorum system's `load` function. The code snippet below is a
shorthand for `grid.strategy(read_fraction=0.25).load(read_fraction=0.25)`.

```python
>>> grid.load(read_fraction=0.25)
0.3749999975
```

The capacity of strategy or quorum is simply the inverse of the load. Our
quorum system has a load of 3/8 on a 25% read workload, so it has a capacity of
8/3.

```python
>>> grid.capacity(read_fraction=0.25)
2.6666666844444444
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
>>> distribution = {0.1: 1, 0.75: 1}
>>> strategy = grid.strategy(read_fraction=distribution)
>>> strategy.load(read_fraction=distribution)
0.40416666474999996
```

## Heterogeneous Node
In the real world, not all nodes are equal. We often run distributed systems on
heterogeneous hardware, so some nodes might be faster than others. To model
this, we instantiate every node with its capacity. Here, nodes `a`, `c`, and
`e` can process 1000 commands per second, while nodes `b`, `d`, and `f` can
only process 500 requests per second.

```python
>>> a = Node('a', capacity=1000)
>>> b = Node('b', capacity=500)
>>> c = Node('c', capacity=1000)
>>> d = Node('d', capacity=500)
>>> e = Node('e', capacity=1000)
>>> f = Node('f', capacity=500)
```

Now, the definition of capacity becomes much simpler. The capacity of a quorum
system is simply the maximum throughput that it can achieve. The load can be
interpreted as the inverse of the capacity. Here, our quorum system is capable
of processing 1333 commands per second for a workload of 75% reads.

```python
>>> grid = QuorumSystem(reads=a*b*c + d*e*f)
>>> strategy = grid.strategy(read_fraction=0.75)
>>> strategy.load(read_fraction=0.75)
0.00075
>>> strategy.capacity(read_fraction=0.75)
1333.3333333333333
```

Nodes might also process reads and writes at different speeds. We can specify
the peak read and write throughput of every node separately. Here, we assume
reads are ten times as fast as writes.

```python
>>> a = Node('a', write_capacity=1000, read_capacity=10000)
>>> b = Node('b', write_capacity=500, read_capacity=5000)
>>> c = Node('c', write_capacity=1000, read_capacity=10000)
>>> d = Node('d', write_capacity=500, read_capacity=5000)
>>> e = Node('e', write_capacity=1000, read_capacity=10000)
>>> f = Node('f', write_capacity=500, read_capacity=5000)
```

With 100% reads, our quorum system can process 10,000 commands per second.
This throughput decreases as we increase the fraction of writes.

```python
>>> grid = QuorumSystem(reads=a*b*c + d*e*f)
>>> grid.capacity(read_fraction=1)
10000.0
>>> grid.capacity(read_fraction=0.5)
3913.043450018904
>>> grid.capacity(read_fraction=0)
2000.0
```

## `f`-resilient Strategies
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
>>> strategy = grid.strategy(read_fraction=0.5, f=1)
```

These sets are quorums even if 1 machine fails.

```python
>>> strategy.get_read_quorum()
{'b', 'f', 'e', 'd', 'a', 'c'}
>>> strategy.get_write_quorum()
{'b', 'd', 'a', 'e'}
```

Note that as we increase resilience, quorums get larger, and we decrease
capacity. On a 100% write workload, our grid quorum system has a 0-resilient
capacity of 2000 commands per second, but a 1-resilient capacity of 1000
commands per second.

```python
>>> grid.capacity(write_fraction=1, f=0)
2000.0
>>> grid.capacity(write_fraction=1, f=1)
1000.0
```

Also note that not all quorum systems are equally as resilient. In the next
code snippet, we construct a "write 2, read 3" quorum system using the `choose`
function. For this quorum system, every set of 2 nodes is a write quorum, and
every set of 3 nodes is a read quorum. This quorum system has a 0-resilient
capacity of 2000 (the same as the grid), but a 1-resilient capacity of 1333
(higher than the grid).

```python
>>> write2 = QuorumSystem(writes=choose(2, [a, b, c, d, e]))
>>> write2.capacity(write_fraction=1, f=0)
2000.0
>>> write2.capacity(write_fraction=1, f=1)
1333.3333333333333
```

## Latency
In the real world, not all nodes are equally as far away. Some are close and
some are far. To address this, we associate every node with a latency, i.e. the
time the required to contact the node. We model this in quoracle by assigning
each node a latency, represented as a `datetime.timedelta`. Here, nodes `a`,
`b`, `c`, `d`, `e`, and `f` in our grid have latencies of 1, 2, 3, 4, 5, and 6
seconds.

```python
>>> import datetime
>>>
>>> def seconds(x: int) -> datetime.timedelta:
>>>     return datetime.timedelta(seconds=x)
>>>
>>> a = Node('a', write_capacity=1000, read_capacity=10000, latency=seconds(1))
>>> b = Node('b', write_capacity=500, read_capacity=5000, latency=seconds(2))
>>> c = Node('c', write_capacity=1000, read_capacity=10000, latency=seconds(3))
>>> d = Node('d', write_capacity=500, read_capacity=5000, latency=seconds(4))
>>> e = Node('e', write_capacity=1000, read_capacity=10000, latency=seconds(5))
>>> f = Node('f', write_capacity=500, read_capacity=5000, latency=seconds(6))
>>> grid = QuorumSystem(reads=a*b*c + d*e*f)
```

The _latency of a quorum_ `q` is the time required to form a quorum of
responses after contacting every node in `q`. For example, the read quorum `{a,
b, c}` has a latency of three seconds. It takes 1 second to hear back from `a`,
another second to hear back from `b`, and then a final second to hear back from
`c`. The write quorum `{a, b, d, f}` has a latency of 4 seconds. It takes 1
second to hear back from `a`, another second to hear back from `b`, and then
another 2 seconds to hear back from `d`. The set `{a, b, d}` is a write quorum,
so the latency of this quorum is 4 seconds. Note that we didn't have to wait to
hear back from `f` in order to form a quorum.

The _latency of a strategy_ is the expected latency of the quorums that it
chooses. The _latency of a quorum system_ is the latency of the latency-optimal
strategy. We can use the `strategy` function to find a latency-optimal strategy
by passing in the value `"latency"` to the `optimize` flag.

```python
>>> sigma = grid.strategy(read_fraction=0.5, optimize='latency')
>>> sigma
Strategy(reads={('a', 'b', 'c'): 1.0}, writes={('c', 'd'): 1.0})
```

We can find the latency of this strategy by calling the `latency` function.

```python
>>> sigma.latency(read_fraction=1)
0:00:03
>>> sigma.latency(read_fraction=0)
0:00:04
>>> sigma.latency(read_fraction=0.5)
0:00:03.500000
```

As with capacity, we can call the `latency` function on our quorum system
directly. In the follow code snippet `grid.latency(read_fraction=0.5,
optimize='latency')` is a shorthand for `grid.strategy(read_fraction=0.5,
optimize='latency').latency(read_fraction=0.5)`.

```
>>> grid.latency(read_fraction=0.5, optimize='latency')
0:00:03.500000
```

Note that finding the latency-optimal strategy is trivial. The latency-optimal
strategy always selects the read and write quorum with the smallest latencies.
However, things get complicated when we start optimizing for capacity and
latency at the same time. When we call the `strategy` function with
`optimize='latency'`, we can pass in a constraint on the maximum allowable load
using the `load_limit` argument. For example, in the code snippet below, we
find the latency-optimal strategy with a capacity of at least 1,500.

```python
>>> sigma = grid.strategy(read_fraction=0.5,
...                      optimize='latency',
...                      load_limit=1/1500)
>>> sigma
Strategy(reads={('a', 'b', 'c'): 1.0}, writes={('a', 'd'): 0.66666667, ('c', 'e'): 0.33333333})
>>> sigma.capacity(read_fraction=0.5)
1499.9999925
>>> sigma.latency(read_fraction=0.5)
0:00:03.666667
```

This strategy always picks the read quorum `{a, b, c}`, and picks the write
quorum `{a, d}` twice as often as write quorum `{c, e}`. It achieves our
desired capacity of 1,500 commands per second (ignoring rounding errors) and
has a latency of 3.66 seconds. We can also find a load-optimal strategy with a
latency constraint.

```python
>>> sigma = grid.strategy(read_fraction=0.5,
...                       optimize='load',
...                       latency_limit=seconds(4))
>>> sigma
Strategy(reads={('a', 'b', 'c'): 0.98870056, ('d', 'e', 'f'): 0.011299435}, writes={('a', 'd'): 0.19548023, ('a', 'f'): 0.22429379, ('b', 'd'): 0.062711864, ('b', 'e'): 0.097740113, ('c', 'e'): 0.41977401})
>>> sigma.capacity(read_fraction=0.5)
3856.2090893331633
>>> sigma.latency(read_fraction=0.5)
0:00:04.000001
```

This strategy is rather complicated and would be hard to find by hand. It has a
capacity of 3856 commands per second and achieves our latency constraint of 4
seconds.

Be careful when specifying constraints. If the constraints cannot be met, a
`NoStrategyFound` exception is raised.

```python
>>> grid.strategy(read_fraction=0.5,
...               optimize='load',
...               latency_limit=seconds(1))
Traceback (most recent call last):
...
quoracle.quorum_system.NoStrategyFoundError: no strategy satisfies the given constraints
```

## Network Load
Another useful metric is network load. When a protocol performs a read, it has
to send messages to every node in a read quorum, and when a protocol performs a
write, it has to send messages to every node in a write quorum. The bigger the
quorums, the more messages are sent over the network. The _network load of a
quorum_ is simply the size of the quorum, the _network load of a strategy_ is
the expected network load of the quorums it chooses, and the _network load of a
quorum system_ is the network load of the network load-optimal strategy.

We can find network load optimal-strategies using the `strategy` function by
passing in `"network"` to the `optimize` flag. We can also specify constraints
on load and latency. In general, using the `strategy` function, we can pick one
of load, latency, or network load to optimize and specify constraints on the
other two metrics.

```python
>>> sigma = grid.strategy(read_fraction=0.5, optimize='network')
>>> sigma
Strategy(reads={('a', 'b', 'c'): 1.0}, writes={('c', 'f'): 1.0})
>>> sigma.network_load(read_fraction=0.5)
2.5
>>> grid.network_load(read_fraction=0.5, optimize='network')
2.5
>>> sigma = grid.strategy(read_fraction=0.5,
...                       optimize='network',
...                       load_limit=1/2000,
...                       latency_limit=seconds(4))
```

## Search
Finding good quorum systems by hand is hard. quoracle includes a heuristic
based search procedure that tries to find quorum systems that are optimal with
respect a target metric and set of constraints. For example, lets try to find a
quorum system

- that has resilience 1,
- that is 1-resilient load optimal for a 75% read workload,
- that has a latency of most 4 seconds, and
- that has a network load of at most 4.

Because the number of quorum systems is enormous, the search procedure can take
a very, very long time. We pass in a timeout to the search procedure to limit
how long it takes. If the timeout expires, `search` returns the most optimal
quorum system that it found so far.

```python
## Search
>>> qs, sigma = search(nodes=[a, b, c, d, e, f],
...                    resilience=1,
...                    f=1,
...                    read_fraction=0.75,
...                    optimize='load',
...                    latency_limit=seconds(4),
...                    network_limit=4,
...                    timeout=seconds(60))
>>> qs
QuorumSystem(reads=choose3(a, c, e, (b + d + f)), writes=choose2(a, c, e, (b * d * f)))
>>> sigma
Strategy(reads={('a', 'c', 'e', 'f'): 0.33333333, ('a', 'b', 'c', 'e'): 0.33333333, ('a', 'c', 'd', 'e'): 0.33333333}, writes={('a', 'b', 'c', 'd', 'f'): 0.15714286, ('b', 'c', 'd', 'e', 'f'): 0.15714286, ('a', 'c', 'e'): 0.52857143, ('a', 'b', 'd', 'e', 'f'): 0.15714286})
>>> sigma.capacity(read_fraction=0.75)
3499.9999536250007
>>> sigma.latency(read_fraction=0.75)
0:00:03.907143
>>> sigma.network_load(read_fraction=0.75)
3.9857142674999997
```

Here, the search procedure returns the quorum system `choose(3, [a, c, e,
b+d+f])` with a capacity of 3500 commands per second and with latency and
network load close to the limits specified.
