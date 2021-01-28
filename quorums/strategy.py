from . import distribution
from .distribution import Distribution
from .expr import Node
from typing import Dict, Generic, List, Optional, Set, TypeVar
import collections
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
                  x: T,
                  read_fraction: Optional[Distribution] = None,
                  write_fraction: Optional[Distribution] = None) \
                  -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(weight * self._node_load(x, fr)
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
        self.plot_node_load_on(ax,
                               nodes=nodes or list(self.nodes),
                               read_fraction=read_fraction,
                               write_fraction=write_fraction)
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
        nodes = nodes or list(self.nodes)
        x_list = [node.x for node in nodes]
        x_index = {x: i for (i, x) in enumerate(x_list)}
        x_ticks = list(range(len(x_list)))
        read_cmap = matplotlib.cm.get_cmap('Reds')
        write_cmap = matplotlib.cm.get_cmap('Blues')
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        fr = sum(weight * fr for (fr, weight) in d.items())
        fw = 1 - fr

        def read_quorum_to_bar_heights(quorum: Set[T]) -> np.array:
            bar_heights = np.zeros(len(x_list))
            for x in quorum:
                bar_heights[x_index[x]] = 1 / self.read_capacity[x]
            return bar_heights

        def write_quorum_to_bar_heights(quorum: Set[T]) -> np.array:
            bar_heights = np.zeros(len(x_list))
            for x in quorum:
                bar_heights[x_index[x]] = 1 / self.write_capacity[x]
            return bar_heights

        bottom = np.zeros(len(x_list))
        for (i, (rq, weight)) in enumerate(zip(self.reads, self.read_weights)):
            bar_heights = fr * weight * read_quorum_to_bar_heights(rq)
            ax.bar(x_ticks,
                   bar_heights,
                   bottom=bottom,
                   color=read_cmap(1 - i * 0.75 / len(self.reads)),
                   edgecolor='white', width=0.8)
            bottom += bar_heights

        for (i, (wq, weight)) in enumerate(zip(self.writes, self.write_weights)):
            bar_heights = fw * weight * write_quorum_to_bar_heights(wq)
            ax.bar(x_ticks,
                   bar_heights,
                   bottom=bottom,
                   color=write_cmap(1 - i * 0.75 / len(self.writes)),
                   edgecolor='white', width=0.8)
            bottom += bar_heights

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(str(x) for x in x_list)

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

    # TODO(mwhittaker): Add read/write load and capacity and read/write cap.
