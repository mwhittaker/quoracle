from . import distribution
from . import geometry
from .distribution import Distribution
from .expr import Node
from .geometry import Point, Segment
from typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar
import collections
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


T = TypeVar('T')


class Strategy(Generic[T]):
    def __init__(self,
                 nodes: Set[Node[T]],
                 reads: List[Set[T]],
                 read_weights: List[float],
                 writes: List[Set[T]],
                 write_weights: List[float]) -> None:
        self.nodes = nodes
        self.read_capacity = {node.x: node.read_capacity for node in nodes}
        self.write_capacity = {node.x: node.write_capacity for node in nodes}
        self.reads = reads
        self.read_weights = read_weights
        self.writes = writes
        self.write_weights = write_weights

        self.unweighted_read_load: Dict[T, float] = \
                collections.defaultdict(float)
        for (read_quorum, weight) in zip(self.reads, self.read_weights):
            for x in read_quorum:
                self.unweighted_read_load[x] += weight

        self.unweighted_write_load: Dict[T, float] = \
                collections.defaultdict(float)
        for (write_quorum, weight) in zip(self.writes, self.write_weights):
            for x in write_quorum:
                self.unweighted_write_load[x] += weight

    def __str__(self) -> str:
        non_zero_reads = {tuple(r): p
                          for (r, p) in zip(self.reads, self.read_weights)
                          if p > 0}
        non_zero_writes = {tuple(w): p
                           for (w, p) in zip(self.writes, self.write_weights)
                           if p > 0}
        return f'Strategy(reads={non_zero_reads}, writes={non_zero_writes})'

    def __repr__(self) -> str:
        return (f'Strategy(nodes={self.nodes}, '+
                         f'reads={self.reads}, ' +
                         f'read_weights={self.read_weights},' +
                         f'writes={self.writes}, ' +
                         f'write_weights={self.write_weights})')

    def get_read_quorum(self) -> Set[T]:
        return np.random.choice(self.reads, p=self.read_weights)

    def get_write_quorum(self) -> Set[T]:
        return np.random.choice(self.writes, p=self.write_weights)

    def load(self,
             read_fraction: Optional[Distribution] = None,
             write_fraction: Optional[Distribution] = None) \
             -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(weight * self._load(fr)
                   for (fr, weight) in d.items())

    def node_load(self,
                  node: Node[T],
                  read_fraction: Optional[Distribution] = None,
                  write_fraction: Optional[Distribution] = None) \
                  -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(weight * self._node_load(node.x, fr)
                   for (fr, weight) in d.items())

    def capacity(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None) \
                 -> float:
        return 1 / self.load(read_fraction, write_fraction)


    def plot_node_load(self,
                       filename: str,
                       nodes: Optional[List[Node[T]]] = None,
                       read_fraction: Optional[Distribution] = None,
                       write_fraction: Optional[Distribution] = None) \
                       -> None:
        fig, ax = plt.subplots()
        self.plot_node_load_on(ax, nodes, read_fraction, write_fraction)
        ax.set_xlabel('Node')
        ax.set_ylabel('Load')
        fig.tight_layout()
        fig.savefig(filename)

    def plot_node_load_on(self,
                          ax: plt.Axes,
                          nodes: Optional[List[Node[T]]] = None,
                          read_fraction: Optional[Distribution] = None,
                          write_fraction: Optional[Distribution] = None) \
                          -> None:
        self._plot_node_load_on(ax,
                                scale=1,
                                scale_by_node_capacity=True,
                                nodes=nodes,
                                read_fraction=read_fraction,
                                write_fraction=write_fraction)

    def plot_node_capacity(self,
                           filename: str,
                           nodes: Optional[List[Node[T]]] = None,
                           read_fraction: Optional[Distribution] = None,
                           write_fraction: Optional[Distribution] = None) \
                           -> None:
        fig, ax = plt.subplots()
        self.plot_node_capacity_on(ax, nodes, read_fraction, write_fraction)
        ax.set_xlabel('Node')
        ax.set_ylabel('Throughput at Peak Throughput')
        fig.tight_layout()
        fig.savefig(filename)

    def plot_node_capacity_on(self,
                              ax: plt.Axes,
                              nodes: Optional[List[Node[T]]] = None,
                              read_fraction: Optional[Distribution] = None,
                              write_fraction: Optional[Distribution] = None) \
                              -> None:
        self._plot_node_load_on(ax,
                                scale=self.capacity(read_fraction,
                                                    write_fraction),
                                scale_by_node_capacity=False,
                                nodes=nodes,
                                read_fraction=read_fraction,
                                write_fraction=write_fraction)

    def plot_node_utilization(self,
                              filename: str,
                              nodes: Optional[List[Node[T]]] = None,
                              read_fraction: Optional[Distribution] = None,
                              write_fraction: Optional[Distribution] = None) \
                              -> None:
        fig, ax = plt.subplots()
        self.plot_node_utilization_on(ax, nodes, read_fraction, write_fraction)
        ax.set_xlabel('Node')
        ax.set_ylabel('Utilization at Peak Throughput')
        fig.tight_layout()
        fig.savefig(filename)

    def plot_node_utilization_on(self,
                                 ax: plt.Axes,
                                 nodes: Optional[List[Node[T]]] = None,
                                 read_fraction: Optional[Distribution] = None,
                                 write_fraction: Optional[Distribution] = None) \
                                 -> None:
        self._plot_node_load_on(
                ax,
                scale=self.capacity(read_fraction, write_fraction),
                scale_by_node_capacity=True,
                nodes=nodes,
                read_fraction=read_fraction,
                write_fraction=write_fraction)

    def plot_load_distribution(self,
                               filename: str,
                               nodes: Optional[List[Node[T]]] = None) \
                               -> None:
        fig, ax = plt.subplots()
        self.plot_load_distribution_on(ax, nodes)
        ax.set_xlabel('Read Fraction')
        ax.set_ylabel('Load')
        fig.tight_layout()
        fig.savefig(filename)

    def _group(self, segments: List[Tuple[Segment, T]]) -> Dict[Segment, List[T]]:
        groups: Dict[Segment, List[T]] = collections.defaultdict(list)
        for segment, x in segments:
            match_found = False
            for other, xs in groups.items():
                if segment.approximately_equal(other):
                    xs.append(x)
                    match_found = True
                    break

            if not match_found:
                groups[segment].append(x)

        return groups

    def plot_load_distribution_on(self,
                                  ax: plt.Axes,
                                  nodes: Optional[List[Node[T]]] = None) \
                                  -> None:
        nodes = nodes or list(self.nodes)

        # We want to plot every node's load distribution. Multiple nodes might
        # have the same load distribution, so we group the nodes by their
        # distribution. The grouping is a little annoying because two floats
        # might not be exactly equal but pretty close.
        groups = self._group([
            (Segment(Point(0, self.node_load(node, read_fraction=0)),
                     Point(1, self.node_load(node, read_fraction=1))), node.x)
            for node in nodes
        ])

        # Compute and plot the max of all segments. We increase the line
        # slightly so it doesn't overlap with the other lines.
        path = geometry.max_of_segments(list(groups.keys()))
        ax.plot([p[0] for p in path],
                [p[1] for p in path],
                label='load',
                linewidth=4)

        for segment, xs in groups.items():
            ax.plot([segment.l.x, segment.r.x],
                    [segment.l.y, segment.r.y],
                    '--',
                    label=','.join(str(x) for x in xs),
                    linewidth=2,
                    alpha=0.75)


    def _node_load(self, x: T, fr: float) -> float:
        """
        _node_load returns the load on x given a fixed read fraction fr.
        """
        fw = 1 - fr
        return (fr * self.unweighted_read_load[x] / self.read_capacity[x] +
                fw * self.unweighted_write_load[x] / self.write_capacity[x])

    def _load(self, fr: float) -> float:
        """
        _load returns the load given a fixed read fraction fr.
        """
        return max(self._node_load(node.x, fr) for node in self.nodes)

    def _plot_node_load_on(
            self,
            ax: plt.Axes,
            scale: float,
            scale_by_node_capacity: bool,
            nodes: Optional[List[Node[T]]] = None,
            read_fraction: Optional[Distribution] = None,
            write_fraction: Optional[Distribution] = None) \
            -> None:
        nodes = nodes or list(self.nodes)
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        x_list = [node.x for node in nodes]
        x_index = {x: i for (i, x) in enumerate(x_list)}
        x_ticks = list(range(len(x_list)))

        def read_quorum_to_bar_heights(quorum: Set[T]) -> np.array:
            bar_heights = np.zeros(len(x_list))
            for x in quorum:
                bar_heights[x_index[x]] = 1
                if scale_by_node_capacity:
                    bar_heights[x_index[x]] /= self.read_capacity[x]
            return bar_heights

        def write_quorum_to_bar_heights(quorum: Set[T]) -> np.array:
            bar_heights = np.zeros(len(x_list))
            for x in quorum:
                bar_heights[x_index[x]] = 1
                if scale_by_node_capacity:
                    bar_heights[x_index[x]] /= self.write_capacity[x]
            return bar_heights

        bottoms = np.zeros(len(x_list))

        fr = sum(weight * fr for (fr, weight) in d.items())
        read_cmap = matplotlib.cm.get_cmap('Reds')
        for (i, (rq, weight)) in enumerate(zip(self.reads, self.read_weights)):
            bar_heights = scale * fr * weight * read_quorum_to_bar_heights(rq)
            ax.bar(x_ticks,
                   bar_heights,
                   bottom=bottoms,
                   color=read_cmap(0.75 - i * 0.5 / len(self.reads)),
                   edgecolor='white', width=0.8)
            for j, (bar_height, bottom) in enumerate(zip(bar_heights, bottoms)):
                if bar_height != 0:
                    ax.text(x_ticks[j], bottom + bar_height / 2, i,
                            ha='center', va='center')
            bottoms += bar_heights

        fw = 1 - fr
        write_cmap = matplotlib.cm.get_cmap('Blues')
        for (i, (wq, weight)) in enumerate(zip(self.writes, self.write_weights)):
            bar_heights = scale * fw * weight * write_quorum_to_bar_heights(wq)
            ax.bar(x_ticks,
                   bar_heights,
                   bottom=bottoms,
                   color=write_cmap(0.75 - i * 0.5 / len(self.writes)),
                   edgecolor='white', width=0.8)
            for j, (bar_height, bottom) in enumerate(zip(bar_heights, bottoms)):
                if bar_height != 0:
                    ax.text(x_ticks[j], bottom + bar_height / 2, i,
                            ha='center', va='center')
            bottoms += bar_heights

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(str(x) for x in x_list)
