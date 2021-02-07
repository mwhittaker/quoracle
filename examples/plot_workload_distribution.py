"""
In this script, we generate a strategy sigma that is optimal for a distribution
of read fractions. We plot this strategy's capacity as a function of read
fraction and compare it to other strategies optimized for specific points in
this distribution. This plot was used in our paper
(https://mwhittaker.github.io/publications/quoracle.pdf).
"""

# See https://stackoverflow.com/a/19521297/3187068
import matplotlib
matplotlib.use('pdf')
font = {'size': 8}
matplotlib.rc('font', **font)

from quoracle import *
import argparse
import itertools
import matplotlib
import matplotlib.pyplot as plt


def main(output_filename: str) -> None:
    a = Node('a', write_capacity=100, read_capacity=200)
    b = Node('b', write_capacity=100, read_capacity=200)
    c = Node('c', write_capacity=50, read_capacity=100)
    d = Node('d', write_capacity=50, read_capacity=100)
    dist = {
        0.00: 10 / 18,
        0.25: 4 / 18,
        0.50: 2 / 18,
        0.75: 1 / 18,
        1.00: 1 / 18,
    }
    qs = QuorumSystem(reads=a*c + b*d)

    xs = [0.01 * i for i in range(101)]
    markers = itertools.cycle(['o', 'v', '^', 'p', '*'])
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    for fr in dist.keys():
        sigma = qs.strategy(read_fraction=fr)
        ys = [sigma.capacity(read_fraction=x) for x in xs]
        ax.plot(xs, ys, '--', label=str(f'$\sigma_{{{fr}}}$'), linewidth=1,
                marker=next(markers), markevery=25, markersize=4, alpha=0.75)

    sigma = qs.strategy(read_fraction=dist)
    ys = [sigma.capacity(read_fraction=x) for x in xs]
    ax.plot(xs, ys, label='$\sigma$', linewidth=1.5,
            marker=next(markers), markevery=25, markersize=4)

    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    ax.set_ylabel('Capacity (commands per second)')
    ax.set_xlabel('Read Fraction')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid()
    fig.tight_layout()
    fig.savefig(output_filename)
    print(f'Wrote figure to "{output_filename}".')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        type=str,
                        default='workload_distribution.pdf',
                        help='Output filename')
    args = parser.parse_args()
    main(args.output)
