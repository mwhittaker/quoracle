from quorums import *
import datetime

def main() -> None:
    def seconds(x: int) -> datetime.timedelta:
        return datetime.timedelta(seconds=x)

    a = Node('a', write_capacity=2000, read_capacity=4000, latency=seconds(1))
    b = Node('b', write_capacity=1000, read_capacity=2000, latency=seconds(1))
    c = Node('c', write_capacity=2000, read_capacity=4000, latency=seconds(3))
    d = Node('d', write_capacity=1000, read_capacity=2000, latency=seconds(4))
    e = Node('e', write_capacity=2000, read_capacity=4000, latency=seconds(5))
    fr = {
        1.00: 0.,
        0.90: 10.,
        0.80: 20.,
        0.70: 100.,
        0.60: 100.,
        0.50: 100.,
        0.40: 60.,
        0.30: 30.,
        0.20: 30.,
        0.10: 20.,
        0.00: 0.,
    }

    maj = QuorumSystem(reads=majority([a, b, c, d, e]))
    grid = QuorumSystem(reads=a*b + c*d*e)
    paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)

    print(maj.capacity(read_fraction=fr))
    print(grid.capacity(read_fraction=fr))
    print(paths.capacity(read_fraction=fr))
    print()

    opt = search(nodes=[a, b, c, d, e],
                 resilience=1,
                 read_fraction=fr)
    sigma = opt.strategy(read_fraction=fr)
    print(opt)
    print(sigma)
    print(sigma.capacity(read_fraction=fr))
    print()

    print(maj)
    print(maj.capacity(read_fraction=fr, f=1))
    print(grid.capacity(read_fraction=fr, f=1))
    print(paths.capacity(read_fraction=fr, f=1))
    print()

    opt = search(nodes=[a, b, c, d, e], resilience=1, read_fraction=fr, f=1)
    sigma = opt.strategy(read_fraction=fr, f=1)
    print(opt)
    print(sigma)
    print(sigma.capacity(read_fraction=fr))
    print()

    print(maj.capacity(read_fraction=fr, optimize='latency', load_limit=1/2000),
          maj.latency(read_fraction=fr, optimize='latency', load_limit=1/2000))
    print(grid.capacity(read_fraction=fr, optimize='latency', load_limit=1/2000),
          grid.latency(read_fraction=fr, optimize='latency', load_limit=1/2000))
    print(paths.capacity(read_fraction=fr, optimize='latency', load_limit=1/2000),
          paths.latency(read_fraction=fr, optimize='latency', load_limit=1/2000))
    print()

    opt = search(nodes=[a, b, c, d, e], resilience=1, read_fraction=fr, optimize='latency', load_limit=1/2000)
    sigma = opt.strategy(read_fraction=fr, optimize='latency', load_limit=1/2000)
    print(opt)
    print(sigma)
    print(sigma.capacity(read_fraction=fr))
    print(sigma.latency(read_fraction=fr))
    print()


if __name__ == '__main__':
    main()
