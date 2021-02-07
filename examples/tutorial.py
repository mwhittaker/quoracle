from quoracle import *


def main() -> None:
    ## Quorum Systems
    a = Node('a')
    b = Node('b')
    c = Node('c')
    d = Node('d')
    e = Node('e')
    f = Node('f')

    grid = QuorumSystem(reads=a*b*c + d*e*f)

    for r in grid.read_quorums():
        print(r)

    for w in grid.write_quorums():
        print(w)

    QuorumSystem(writes=(a + b + c) * (d + e + f))

    QuorumSystem(reads=a*b*c + d*e*f, writes=(a + b + c) * (d + e + f))

    # QuorumSystem(reads=a+b+c, writes=d+e+f)
    # ValueError: Not all read quorums intersect all write quorums

    print(grid.is_read_quorum({'a', 'b', 'c'}))       # True
    print(grid.is_read_quorum({'a', 'b', 'c', 'd'}))  # True
    print(grid.is_read_quorum({'a', 'b', 'd'}))       # False

    print(grid.is_write_quorum({'a', 'd'}))      # True
    print(grid.is_write_quorum({'a', 'd', 'd'})) # True
    print(grid.is_write_quorum({'a', 'b'}))      # False

    ## Resilience
    print(grid.read_resilience())  # 1
    print(grid.write_resilience()) # 2
    print(grid.resilience())       # 1

    ## Strategies
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

    print(strategy.get_read_quorum())
    print(strategy.get_read_quorum())
    print(strategy.get_read_quorum())
    print(strategy.get_read_quorum())
    print(strategy.get_write_quorum())
    print(strategy.get_write_quorum())
    print(strategy.get_write_quorum())
    print(strategy.get_write_quorum())

    ## Load and Capacity
    print(strategy.load(read_fraction=1)) # 2/3

    print(strategy.load(write_fraction=1)) # 1/3

    print(strategy.load(read_fraction=0.25)) # 5/12

    print(strategy.node_load(a, read_fraction=0.25)) # 5/12
    print(strategy.node_load(b, read_fraction=0.25)) # 5/12
    print(strategy.node_load(c, read_fraction=0.25)) # 5/12
    print(strategy.node_load(d, read_fraction=0.25)) # 1/3
    print(strategy.node_load(e, read_fraction=0.25)) # 1/3
    print(strategy.node_load(f, read_fraction=0.25)) # 1/3

    strategy = grid.strategy(read_fraction=0.25)
    print(strategy)
    print(strategy.load(read_fraction=0.25)) # 3/8

    print(strategy.load(read_fraction=0))   # 1/3
    print(strategy.load(read_fraction=0.5)) # 5/12
    print(strategy.load(read_fraction=1))   # 1/2

    print(grid.load(read_fraction=0.25)) # 3/8

    print(grid.capacity(read_fraction=0.25)) # 8/3

    ## Workload Distributions
    distribution = {0.1: 0.5, 0.75: 0.5}
    strategy = grid.strategy(read_fraction=distribution)
    print(strategy.load(read_fraction=distribution)) # 0.404

    ## Heterogeneous Node
    a = Node('a', capacity=1000)
    b = Node('b', capacity=500)
    c = Node('c', capacity=1000)
    d = Node('d', capacity=500)
    e = Node('e', capacity=1000)
    f = Node('f', capacity=500)

    grid = QuorumSystem(reads=a*b*c + d*e*f)
    strategy = grid.strategy(read_fraction=0.75)
    print(strategy.load(read_fraction=0.75))     # 0.00075
    print(strategy.capacity(read_fraction=0.75)) # 1333

    a = Node('a', write_capacity=1000, read_capacity=10000)
    b = Node('b', write_capacity=500, read_capacity=5000)
    c = Node('c', write_capacity=1000, read_capacity=10000)
    d = Node('d', write_capacity=500, read_capacity=5000)
    e = Node('e', write_capacity=1000, read_capacity=10000)
    f = Node('f', write_capacity=500, read_capacity=5000)

    grid = QuorumSystem(reads=a*b*c + d*e*f)
    print(grid.capacity(read_fraction=1))   # 10,000
    print(grid.capacity(read_fraction=0.5)) # 3913
    print(grid.capacity(read_fraction=0))   # 2000

    ## f-resilient Strategies
    strategy = grid.strategy(read_fraction=0.5, f=1)

    print(strategy.get_read_quorum())
    print(strategy.get_write_quorum())

    print(grid.capacity(write_fraction=1, f=0))
    print(grid.capacity(write_fraction=1, f=1))

    write2 = QuorumSystem(writes=choose(2, [a, b, c, d, e]))
    print(write2.capacity(write_fraction=1, f=0))
    print(write2.capacity(write_fraction=1, f=1))

    ## Latency
    import datetime

    def seconds(x: int) -> datetime.timedelta:
        return datetime.timedelta(seconds=x)

    a = Node('a', write_capacity=1000, read_capacity=10000, latency=seconds(1))
    b = Node('b', write_capacity=500, read_capacity=5000, latency=seconds(2))
    c = Node('c', write_capacity=1000, read_capacity=10000, latency=seconds(3))
    d = Node('d', write_capacity=500, read_capacity=5000, latency=seconds(4))
    e = Node('e', write_capacity=1000, read_capacity=10000, latency=seconds(5))
    f = Node('f', write_capacity=500, read_capacity=5000, latency=seconds(6))
    grid = QuorumSystem(reads=a*b*c + d*e*f)

    sigma = grid.strategy(read_fraction=0.5, optimize='latency')
    print(sigma)

    print(sigma.latency(read_fraction=1))
    print(sigma.latency(read_fraction=0))
    print(sigma.latency(read_fraction=0.5))

    print(grid.latency(read_fraction=0.5, optimize='latency'))

    sigma = grid.strategy(read_fraction=0.5,
                          optimize='latency',
                          load_limit=1/1500)
    print(sigma)
    print(sigma.capacity(read_fraction=0.5))
    print(sigma.latency(read_fraction=0.5))

    sigma = grid.strategy(read_fraction=0.5,
                          optimize='load',
                          latency_limit=seconds(4))
    print(sigma)
    print(sigma.capacity(read_fraction=0.5))
    print(sigma.latency(read_fraction=0.5))

    # grid.strategy(read_fraction=0.5,
    #               optimize='load',
    #               latency_limit=seconds(1))
    # quoracle.quorum_system.NoStrategyFoundError: no strategy satisfies the given constraints

    ## Network Load
    sigma = grid.strategy(read_fraction=0.5, optimize='network')
    print(sigma)
    print(sigma.network_load(read_fraction=0.5))
    print(grid.network_load(read_fraction=0.5, optimize='network'))
    sigma = grid.strategy(read_fraction=0.5,
                          optimize='network',
                          load_limit=1/2000,
                          latency_limit=seconds(4))

    ## Search
    qs, sigma = search(nodes=[a, b, c, d, e, f],
                       resilience=1,
                       f=1,
                       read_fraction=0.75,
                       optimize='load',
                       latency_limit=seconds(4),
                       network_limit=4,
                       timeout=seconds(60))
    print(qs)
    print(sigma)
    print(sigma.capacity(read_fraction=0.75))
    print(sigma.latency(read_fraction=0.75))
    print(sigma.network_load(read_fraction=0.75))


if __name__ == '__main__':
    main()
