from . import distribution
from .distribution import Distribution
from .expr import Node
from .strategy import Strategy
from typing import List, Optional, Set, TypeVar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


T = TypeVar('T')


def plot_node_load(filename: str,
                   strategy: Strategy[T],
                   nodes: Optional[List[Node[T]]] = None,
                   read_fraction: Optional[Distribution] = None,
                   write_fraction: Optional[Distribution] = None):
    fig, ax = plt.subplots()
    plot_node_load_on(ax, strategy, nodes, read_fraction, write_fraction)
    ax.set_xlabel('Node')
    ax.set_ylabel('Load')
    fig.tight_layout()
    fig.savefig(filename)


def plot_node_load_on(ax: plt.Axes,
                      strategy: Strategy[T],
                      nodes: Optional[List[Node[T]]] = None,
                      read_fraction: Optional[Distribution] = None,
                      write_fraction: Optional[Distribution] = None):
    _plot_node_load_on(ax,
                       strategy,
                       nodes or list(strategy.nodes),
                       scale=1,
                       scale_by_node_capacity=True,
                       read_fraction=read_fraction,
                       write_fraction=write_fraction)


def plot_node_utilization(filename: str,
                          strategy: Strategy[T],
                          nodes: Optional[List[Node[T]]] = None,
                          read_fraction: Optional[Distribution] = None,
                          write_fraction: Optional[Distribution] = None):
    fig, ax = plt.subplots()
    plot_node_utilization_on(ax, strategy, nodes, read_fraction, write_fraction)
    ax.set_xlabel('Node')
    ax.set_ylabel('Utilization')
    fig.tight_layout()
    fig.savefig(filename)


def plot_node_utilization_on(ax: plt.Axes,
                      strategy: Strategy[T],
                      nodes: Optional[List[Node[T]]] = None,
                      read_fraction: Optional[Distribution] = None,
                      write_fraction: Optional[Distribution] = None):
    _plot_node_load_on(ax,
                       strategy,
                       nodes or list(strategy.nodes),
                       scale=strategy.capacity(read_fraction, write_fraction),
                       scale_by_node_capacity=True,
                       read_fraction=read_fraction,
                       write_fraction=write_fraction)


def plot_node_throughput(filename: str,
                   strategy: Strategy[T],
                   nodes: Optional[List[Node[T]]] = None,
                   read_fraction: Optional[Distribution] = None,
                   write_fraction: Optional[Distribution] = None):
    fig, ax = plt.subplots()
    plot_node_throughput_on(ax, strategy, nodes, read_fraction, write_fraction)
    ax.set_xlabel('Node')
    ax.set_ylabel('Throughput')
    fig.tight_layout()
    fig.savefig(filename)


def plot_node_throughput_on(ax: plt.Axes,
                            strategy: Strategy[T],
                            nodes: Optional[List[Node[T]]] = None,
                            read_fraction: Optional[Distribution] = None,
                            write_fraction: Optional[Distribution] = None):
    _plot_node_load_on(ax,
                       strategy,
                       nodes or list(strategy.nodes),
                       scale=strategy.capacity(read_fraction, write_fraction),
                       scale_by_node_capacity=False,
                       read_fraction=read_fraction,
                       write_fraction=write_fraction)


def _plot_node_load_on(ax: plt.Axes,
                       sigma: Strategy[T],
                       nodes: List[Node[T]],
                       scale: float,
                       scale_by_node_capacity: bool,
                       read_fraction: Optional[Distribution] = None,
                       write_fraction: Optional[Distribution] = None):
    d = distribution.canonicalize_rw(read_fraction, write_fraction)
    x_list = [node.x for node in nodes]
    x_index = {x: i for (i, x) in enumerate(x_list)}
    x_ticks = list(range(len(x_list)))

    def one_hot(quorum: Set[T]) -> np.array:
        bar_heights = np.zeros(len(x_list))
        for x in quorum:
            bar_heights[x_index[x]] = 1
        return bar_heights

    def plot_quorums(quorums: List[Set[T]],
                     weights: List[float],
                     fraction: float,
                     bottoms: np.array,
                     capacities: np.array,
                     cmap: matplotlib.colors.Colormap):
        for (i, (quorum, weight)) in enumerate(zip(quorums, weights)):
            bar_heights = scale * fraction * weight * one_hot(quorum)
            if scale_by_node_capacity:
                bar_heights /= capacities

            ax.bar(x_ticks,
                   bar_heights,
                   bottom=bottoms,
                   color=cmap(0.75 - i * 0.5 / len(quorums)),
                   edgecolor='white', width=0.8)

            for j, (bar_height, bottom) in enumerate(zip(bar_heights, bottoms)):
                # TODO(mwhittaker): Fix the unhappy typechecker.
                text = ''.join(str(x) for x in sorted(list(quorum)))
                if bar_height != 0:
                    ax.text(x_ticks[j], bottom + bar_height / 2, text,
                            ha='center', va='center')
            bottoms += bar_heights

    fr = sum(weight * fr for (fr, weight) in d.items())
    fw = 1 - fr
    read_capacities = np.array([node.read_capacity for node in nodes])
    write_capacities = np.array([node.write_capacity for node in nodes])
    bottoms = np.zeros(len(x_list))
    plot_quorums(sigma.reads, sigma.read_weights, fr, bottoms, read_capacities,
                 matplotlib.cm.get_cmap('Reds'))
    plot_quorums(sigma.writes, sigma.write_weights, fw, bottoms,
                 write_capacities, matplotlib.cm.get_cmap('Blues'))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(str(x) for x in x_list)
