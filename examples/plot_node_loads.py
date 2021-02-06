from quoracle import *
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main():
    a = Node('a', write_capacity=1000, read_capacity=10000)
    b = Node('b', write_capacity=500, read_capacity=5000)
    c = Node('c', write_capacity=1000, read_capacity=10000)
    d = Node('d', write_capacity=500, read_capacity=5000)
    e = Node('e', write_capacity=1000, read_capacity=10000)
    fr = 0.9
    nodes = [a, b, c, d, e]

    simple_majority = QuorumSystem(reads=majority([a, b, c, d, e]))
    crumbling_walls = QuorumSystem(reads=a*b + c*d*e)
    paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)
    opt, _ = search(nodes, read_fraction=fr, timeout=datetime.timedelta(seconds=9))

    fig, ax = plt.subplots(3, 4, figsize = (6.3 * 2, 4.8 * 2), sharey='row')
    for i, qs in enumerate([simple_majority, crumbling_walls, paths, opt]):
        sigma = qs.strategy(read_fraction=fr)
        print(qs, sigma.capacity(read_fraction=fr))
        plot_node_load_on(ax[0][i], sigma, nodes=nodes, read_fraction=fr)
        plot_node_utilization_on(ax[1][i], sigma, nodes=nodes, read_fraction=fr)
        plot_node_throughput_on(ax[2][i], sigma, nodes=nodes, read_fraction=fr)

    ax[0][0].set_title('Simple Majority')
    ax[0][1].set_title('Crumbling Walls')
    ax[0][2].set_title('Paths')
    ax[0][3].set_title(f'Opt {opt.reads}')
    ax[0][0].set_ylabel('Load')
    ax[1][0].set_ylabel('Utilization at Peak Throughput')
    ax[2][0].set_ylabel('Throughput at Peak Throughput')
    fig.tight_layout()
    fig.savefig('node_loads.pdf')


if __name__ == '__main__':
    main()
