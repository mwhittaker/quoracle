from quorums import *
import datetime


def main() -> None:
    # The basics.
    a, b, c = Node('a'), Node('b'), Node('c')
    majority = QuorumSystem(reads=a*b + b*c + a*c)
    print(majority.resilience()) # 1
    print(majority.capacity(read_fraction=1)) # 1.5
    print()

    # Different capacities
    a, b = Node('a', capacity=100), Node('b', capacity=100)
    c, d = Node('c', capacity=50), Node('d', capacity=50)
    grid = QuorumSystem(reads=a*b + c*d)
    print(grid.capacity(read_fraction=1)) # 150
    print()

    # Different read and write capacities.
    a = Node('a', write_capacity=100, read_capacity=200)
    b = Node('b', write_capacity=100, read_capacity=200)
    c = Node('c', write_capacity=50, read_capacity=100)
    d = Node('d', write_capacity=50, read_capacity=100)
    grid = QuorumSystem(reads=a*b + c*d)
    print(grid.capacity(read_fraction=1)) # 300
    print(grid.capacity(read_fraction=0.5)) # 200
    print(grid.capacity(read_fraction=0)) # 100
    print()

    # Workload distribution.
    grid = QuorumSystem(reads=a*c + b*d)
    fr = {0.00: 10 / 18,
          0.25: 4 / 18,
          0.50: 2 / 18,
          0.75: 1 / 18,
          1.00: 1 / 18}
    print(grid.capacity(read_fraction=fr)) # 159
    print()

    # f-resilient strategies.
    grid = QuorumSystem(reads=a*b + c*d)
    choose2 = QuorumSystem(reads=choose(2, [a, b, c, d]))
    print(grid.capacity(read_fraction=1)) # 300
    print(choose2.capacity(read_fraction=1)) # 300
    print(grid.capacity(read_fraction=1, f=1)) # 100
    print(choose2.capacity(read_fraction=1, f=1)) # 200
    print(choose2.strategy(read_fraction=1, f=1)) # abd: 0.5, abc: 0.5
    print()

    # Network load and latency.
    def seconds(n: int) -> datetime.timedelta:
        return datetime.timedelta(seconds=n)

    a = Node('a', write_capacity=100, read_capacity=200, latency=seconds(4))
    b = Node('b', write_capacity=100, read_capacity=200, latency=seconds(4))
    c = Node('c', write_capacity=50, read_capacity=100, latency=seconds(1))
    d = Node('d', write_capacity=50, read_capacity=100, latency=seconds(1))
    grid = QuorumSystem(reads=a*b + c*d)
    sigma = grid.strategy(read_fraction=1)
    print(sigma.capacity(read_fraction=1)) # 300
    print(sigma.network_load(read_fraction=1)) # 2
    print(sigma.latency(read_fraction=1)) # 3 seconds
    print(grid.strategy(
        read_fraction = 1,
        optimize = 'latency',
        load_limit = 1 / 150,
        network_limit = 2,
    )) # ab: 1/3, cd: 2/3
    print()

    # Search.
    qs = search(
        nodes = [a, b, c, d],
        read_fraction = 1,
        optimize = 'latency',
        load_limit = 1 / 150,
        network_limit = 2,
        # timeout = seconds(3),
    )
    print(qs) # a + b + c + d
    sigma = qs.strategy(
        read_fraction = 1,
        optimize = 'latency',
        load_limit = 1 / 150,
        network_limit = 2,
    )
    print(sigma) # c: 1/3 d: 2/3
    print(sigma.capacity(read_fraction=1)) # 150
    print(sigma.network_load(read_fraction=1)) # 1
    print(sigma.latency(read_fraction=1)) # 1


if __name__ == '__main__':
    main()
