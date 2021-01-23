from . import distribution
from .distribution import Distribution
from .expr import Node
from typing import Dict, Generic, List, Optional, Set, TypeVar
import collections
import numpy as np


T = TypeVar('T')


class Strategy(Generic[T]):
    def load(self,
             read_fraction: Optional[Distribution] = None,
             write_fraction: Optional[Distribution] = None) \
             -> float:
        raise NotImplementedError

    def capacity(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None) \
                 -> float:
        return 1 / self.load(read_fraction, write_fraction)

    def get_read_quorum(self) -> Set[T]:
        raise NotImplementedError

    def get_write_quorum(self) -> Set[T]:
        raise NotImplementedError


class ExplicitStrategy(Strategy[T]):
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

    def __str__(self) -> str:
        non_zero_reads = {tuple(r): p
                          for (r, p) in zip(self.reads, self.read_weights)
                          if p > 0}
        non_zero_writes = {tuple(w): p
                           for (w, p) in zip(self.writes, self.write_weights)
                           if p > 0}
        return (f'ExplicitStrategy(reads={non_zero_reads}, ' +
                                 f'writes={non_zero_writes})')

    def __repr__(self) -> str:
        return (f'ExplicitStrategy(nodes={self.nodes}, '+
                                 f'reads={self.reads}, ' +
                                 f'read_weights={self.read_weights},' +
                                 f'writes={self.writes}, ' +
                                 f'write_weights={self.write_weights})')

    def load(self,
             read_fraction: Optional[Distribution] = None,
             write_fraction: Optional[Distribution] = None) \
             -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        fr = sum(f * weight for (f, weight) in d.items())

        read_load: Dict[T, float] = collections.defaultdict(float)
        for (read_quorum, weight) in zip(self.reads, self.read_weights):
            for x in read_quorum:
                read_load[x] += weight

        write_load: Dict[T, float] = collections.defaultdict(float)
        for (write_quorum, weight) in zip(self.writes, self.write_weights):
            for x in write_quorum:
                write_load[x] += weight

        loads: List[float] = []
        for node in self.nodes:
            x = node.x
            load = 0.0
            if x in read_load:
                load += fr * read_load[x] / self.read_capacity[x]
            if x in write_load:
                load += (1 - fr) * write_load[x] / self.write_capacity[x]
            loads.append(load)

        return max(loads)

    # TODO(mwhittaker): Add read/write load and capacity and read/write cap.

    def get_read_quorum(self) -> Set[T]:
        return np.random.choice(self.reads, p=self.read_weights)

    def get_write_quorum(self) -> Set[T]:
        return np.random.choice(self.writes, p=self.write_weights)
