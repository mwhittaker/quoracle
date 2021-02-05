from . import distribution
from . import geometry
from .distribution import Distribution
from .expr import Node
from .geometry import Point, Segment
from .quorum_system import Strategy
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, TypeVar
import collections
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
                       nodes or list(strategy.nodes()),
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
                       nodes or list(strategy.nodes()),
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
                            write_fraction: Optional[Distribution] = None,
                            draw_node_capacities: bool = True):
    nodes = nodes or list(strategy.nodes())
    d = distribution.canonicalize_rw(read_fraction, write_fraction)
    fr = sum(weight * fr for (fr, weight) in d.items())
    fw = 1 - fr
    # TODO(mwhittaker): Explain.
    node_limits = [
        fr * strategy.x_read_probability[node.x] / node_load +
        fw * strategy.x_write_probability[node.x] / node_load
        for node in nodes
        for node_load in [strategy.node_load(
            node,
            read_fraction=read_fraction,
            write_fraction=write_fraction
        )]
    ]
    _plot_node_load_on(ax,
                       strategy,
                       nodes,
                       scale=strategy.capacity(read_fraction, write_fraction),
                       scale_by_node_capacity=False,
                       read_fraction=read_fraction,
                       write_fraction=write_fraction,
                       node_limits=node_limits if draw_node_capacities else None)


def _plot_node_load_on(ax: plt.Axes,
                       sigma: Strategy[T],
                       nodes: List[Node[T]],
                       scale: float,
                       scale_by_node_capacity: bool,
                       read_fraction: Optional[Distribution] = None,
                       write_fraction: Optional[Distribution] = None,
                       node_limits: List[float] = None):
    d = distribution.canonicalize_rw(read_fraction, write_fraction)
    x_list = [node.x for node in nodes]
    x_index = {x: i for (i, x) in enumerate(x_list)}
    x_ticks = list(range(len(x_list)))

    def one_hot(quorum: FrozenSet[T]) -> np.array:
        bar_heights = np.zeros(len(x_list))
        for x in quorum:
            bar_heights[x_index[x]] = 1
        return bar_heights

    width = 0.8
    def plot_quorums(sigma: Dict[FrozenSet[T], float],
                     fraction: float,
                     bottoms: np.array,
                     capacities: np.array,
                     cmap: matplotlib.colors.Colormap):
        for (i, (quorum, weight)) in enumerate(sigma.items()):
            bar_heights = scale * fraction * weight * one_hot(quorum)
            if scale_by_node_capacity:
                bar_heights /= capacities

            ax.bar(x_ticks,
                   bar_heights,
                   bottom=bottoms,
                   color=cmap(0.75 - i * 0.5 / len(sigma)),
                   edgecolor='white', width=width)

            for j, (bar_height, bottom) in enumerate(zip(bar_heights, bottoms)):
                text = ''.join(str(x) for x in sorted(list(quorum))) # type: ignore
                if bar_height != 0:
                    ax.text(x_ticks[j], bottom + bar_height / 2, text,
                            ha='center', va='center')
            bottoms += bar_heights

    # Plot the quorums.
    fr = sum(weight * fr for (fr, weight) in d.items())
    fw = 1 - fr
    read_capacities = np.array([node.read_capacity for node in nodes])
    write_capacities = np.array([node.write_capacity for node in nodes])
    bottoms = np.zeros(len(x_list))
    plot_quorums(sigma.sigma_r, fr, bottoms, read_capacities,
                 matplotlib.cm.get_cmap('Reds'))
    plot_quorums(sigma.sigma_w, fw, bottoms,
                 write_capacities, matplotlib.cm.get_cmap('Blues'))

    # Plot the limits, if there are any.
    if node_limits is not None:
        for (i, limit) in enumerate(node_limits):
            ax.plot([i - width/2, i + width/2], [limit, limit], color='black')

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(str(x) for x in x_list)


def plot_load_distribution(filename: str,
                           strategy: Strategy[T],
                           nodes: Optional[List[Node[T]]] = None):
    fig, ax = plt.subplots()
    plot_load_distribution_on(ax, strategy, nodes)
    ax.set_xlabel('Read Fraction')
    ax.set_ylabel('Load')
    fig.tight_layout()
    fig.savefig(filename)


def _group(segments: Dict[T, Segment]) -> Dict[Segment, List[T]]:
    groups: Dict[Segment, List[T]] = collections.defaultdict(list)
    for x, segment in segments.items():
        matches = (s for s in groups if segment.approximately_equal(s))
        groups[next(matches, segment)].append(x)
    return groups


def plot_load_distribution_on(ax: plt.Axes,
                              strategy: Strategy[T],
                              nodes: Optional[List[Node[T]]] = None):
    nodes = nodes or list(strategy.nodes())

    # We want to plot every node's load distribution. Multiple nodes might
    # have the same load distribution, so we group the nodes by their
    # distribution. The grouping is a little annoying because two floats
    # might not be exactly equal but pretty close.
    groups = _group({
        node.x: Segment(
            Point(0, strategy.node_load(node, read_fraction=0)),
            Point(1, strategy.node_load(node, read_fraction=1))
        )
        for node in nodes
    })

    # Compute and plot the max of all segments. We plot the load first so that
    # it lies underneath the node loads.
    path = geometry.max_of_segments(list(groups.keys()))
    ax.plot([p[0] for p in path],
            [p[1] for p in path],
            label='load',
            linewidth=4)

    # We plot the node loads second so that they appear above the load.
    for segment, xs in groups.items():
        ax.plot([segment.l.x, segment.r.x],
                [segment.l.y, segment.r.y],
                '--',
                label=','.join(str(x) for x in xs),
                linewidth=2,
                alpha=0.75)
