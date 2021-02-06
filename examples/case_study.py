# See https://stackoverflow.com/a/19521297/3187068
import matplotlib
matplotlib.use('pdf')
font = {'size': 8}
matplotlib.rc('font', **font)

from quoracle import *
import datetime
import matplotlib.pyplot as plt

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

    print('0-resilient Capacities')
    print(maj.uniform_strategy().capacity(read_fraction=fr))
    print(maj.capacity(read_fraction=fr))
    print(grid.capacity(read_fraction=fr))
    print(paths.capacity(read_fraction=fr))
    print()

    print('0-resilient Searched')
    start = datetime.datetime.now()
    opt, _ = search(nodes=[a, b, c, d, e],
                    resilience=1,
                    read_fraction=fr)
    stop = datetime.datetime.now()
    print((stop - start))
    sigma = opt.strategy(read_fraction=fr)
    print(opt)
    print(sigma)
    print(sigma.capacity(read_fraction=fr))
    print()

    for (sigma, name, filename, size) in [
        (maj.uniform_strategy(),
         'Majority Quorum System',
         'majority_uniform',
         (3.25, 2)),
        (sigma,
         'Searched Quorum System',
         'searched',
         (3.25, 1.75)),
    ]:
        fig, ax = plt.subplots(figsize=size)
        plot_node_throughput_on(
            ax,
            sigma,
            nodes = [a, b, c, d, e],
            read_fraction=0.5,
            draw_node_capacities=False,
        )
        ax.set_xlabel('Node')
        ax.set_ylabel('Throughput\n(commands per second)')
        fig.tight_layout()
        fig.savefig(f'{filename}_throughputs.pdf')

    print('1-resilient Capacities')
    print(maj)
    print(maj.capacity(read_fraction=fr, f=1))
    print(grid.capacity(read_fraction=fr, f=1))
    print(paths.capacity(read_fraction=fr, f=1))
    print()

    print('1-resilient Searched')
    start = datetime.datetime.now()
    opt, _ = search(nodes=[a, b, c, d, e], resilience=1, read_fraction=fr, f=1)
    stop = datetime.datetime.now()
    print(stop - start)
    sigma = opt.strategy(read_fraction=fr, f=1)
    print(opt)
    print(sigma)
    print(sigma.capacity(read_fraction=fr))
    print()

    print('Latency Optimal Capacities and Latencies')
    print(maj.uniform_strategy().capacity(read_fraction=fr),
          maj.uniform_strategy().latency(read_fraction=fr))
    print(maj.capacity(read_fraction=fr, optimize='latency', load_limit=1/2000),
          maj.latency(read_fraction=fr, optimize='latency', load_limit=1/2000))
    print(grid.capacity(read_fraction=fr, optimize='latency', load_limit=1/2000),
          grid.latency(read_fraction=fr, optimize='latency', load_limit=1/2000))
    print(paths.capacity(read_fraction=fr, optimize='latency', load_limit=1/2000),
          paths.latency(read_fraction=fr, optimize='latency', load_limit=1/2000))
    print()

    print('Latency Optimal Searched')
    start = datetime.datetime.now()
    opt, _ = search(nodes=[a, b, c, d, e], resilience=1, read_fraction=fr,
                 optimize='latency', load_limit=1/2000)
    stop = datetime.datetime.now()
    print(stop - start)
    sigma = opt.strategy(read_fraction=fr, optimize='latency', load_limit=1/2000)
    print(opt)
    print(sigma)
    print(sigma.capacity(read_fraction=fr))
    print(sigma.latency(read_fraction=fr))
    print()


if __name__ == '__main__':
    main()
