# TODO(mwhittaker): We can define a set of read quorums that are not minimal.
# Does this mess things up?

from . import distribution
from .distribution import Distribution
from .expr import Expr, Node
from .strategy import ExplicitStrategy, Strategy
from typing import Dict, Iterator, Generic, List, Optional, Set, TypeVar
import collections
import itertools
import pulp


T = TypeVar('T')


class QuorumSystem(Generic[T]):
    def __init__(self, reads: Optional[Expr[T]] = None,
                       writes: Optional[Expr[T]] = None) -> None:
        if reads is not None and writes is not None:
            optimal_writes = reads.dual()
            if not all(optimal_writes.is_quorum(write_quorum)
                       for write_quorum in writes.quorums()):
                raise ValueError(
                    'Not all read quorums intersect all write quorums')

            self.reads = reads
            self.writes = writes
        elif reads is not None and writes is None:
            self.reads = reads
            self.writes = reads.dual()
        elif reads is None and writes is not None:
            self.reads = writes.dual()
            self.writes = writes
        else:
            raise ValueError('A QuorumSystem must be instantiated with a set '
                             'of read quorums or a set of write quorums')

    def __repr__(self) -> str:
        return f'QuorumSystem(reads={self.reads}, writes={self.writes})'

    def read_quorums(self) -> Iterator[Set[T]]:
        return self.reads.quorums()

    def write_quorums(self) -> Iterator[Set[T]]:
        return self.writes.quorums()

    def is_read_quorum(self, xs: Set[T]) -> bool:
        return self.reads.is_quorum(xs)

    def is_write_quorum(self, xs: Set[T]) -> bool:
        return self.writes.is_quorum(xs)

    def nodes(self) -> Set[Node[T]]:
        return self.reads.nodes() | self.writes.nodes()

    def resilience(self) -> int:
        return min(self.read_resilience(), self.write_resilience())

    def read_resilience(self) -> int:
        return self.reads.resilience()

    def write_resilience(self) -> int:
        return self.writes.resilience()

    def strategy(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None,
                 f: int = 0) \
                 -> 'Strategy[T]':
        if f < 0:
            raise ValueError('f must be >= 0')

        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        if f == 0:
            return self._load_optimal_strategy(
                list(self.read_quorums()),
                list(self.write_quorums()),
                d)
        else:
            xs = [node.x for node in self.nodes()]
            read_quorums = list(self._f_resilient_quorums(f, xs, self.reads))
            write_quorums = list(self._f_resilient_quorums(f, xs, self.reads))
            if len(read_quorums) == 0:
                raise ValueError(f'There are no {f}-resilient read quorums')
            if len(write_quorums) == 0:
                raise ValueError(f'There are no {f}-resilient write quorums')
            return self._load_optimal_strategy(read_quorums, write_quorums, d)

    def dup_free(self) -> bool:
        return self.reads.dup_free() and self.writes.dup_free()

    def _f_resilient_quorums(self,
                             f: int,
                             xs: List[T],
                             e: Expr) -> Iterator[Set[T]]:
        assert f >= 1

        def helper(s: Set[T], i: int) -> Iterator[Set[T]]:
            if all(e.is_quorum(s - set(failure))
                   for failure in itertools.combinations(s, min(f, len(s)))):
                yield set(s)
                return

            for j in range(i, len(xs)):
                s.add(xs[j])
                yield from helper(s, j + 1)
                s.remove(xs[j])

        return helper(set(), 0)

    def load(self,
             read_fraction: Optional[Distribution] = None,
             write_fraction: Optional[Distribution] = None,
             f: int = 0) \
             -> float:
        sigma = self.strategy(read_fraction, write_fraction, f)
        return sigma.load(read_fraction, write_fraction)

    def capacity(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None,
                 f: int = 0) \
                 -> float:
        return 1 / self.load(read_fraction, write_fraction, f)

    def _load_optimal_strategy(self,
                               read_quorums: List[Set[T]],
                               write_quorums: List[Set[T]],
                               read_fraction: Dict[float, float]) \
                               -> 'Strategy[T]':
        # TODO(mwhittaker): Explain f_r calculation.
        fr = sum(f * weight for (f, weight) in read_fraction.items())

        nodes = self.reads.nodes() | self.writes.nodes()
        read_capacity = {node.x: node.read_capacity for node in nodes}
        write_capacity = {node.x: node.write_capacity for node in nodes}

        read_quorum_vars: List[pulp.LpVariable] = []
        x_to_read_quorum_vars: Dict[T, List[pulp.LpVariable]] = \
            collections.defaultdict(list)

        for (i, read_quorum) in enumerate(read_quorums):
            v = pulp.LpVariable(f'r{i}', 0, 1)
            read_quorum_vars.append(v)
            for x in read_quorum:
                x_to_read_quorum_vars[x].append(v)

        write_quorum_vars: List[pulp.LpVariable] = []
        x_to_write_quorum_vars: Dict[T, List[pulp.LpVariable]] = \
            collections.defaultdict(list)
        for (i, write_quorum) in enumerate(write_quorums):
            v = pulp.LpVariable(f'w{i}', 0, 1)
            write_quorum_vars.append(v)
            for x in write_quorum:
                x_to_write_quorum_vars[x].append(v)

        # Form the linear program to find the load.
        problem = pulp.LpProblem("load", pulp.LpMinimize)

        # If we're trying to balance the strategy, then we want to minimize the
        # pairwise absolute differences between the read probabilities and the
        # write probabilities.
        l = pulp.LpVariable('l', 0, 1)
        problem += l
        problem += (sum(read_quorum_vars) == 1, 'valid read strategy')
        problem += (sum(write_quorum_vars) == 1, 'valid write strategy')
        for node in nodes:
            x = node.x
            x_load: pulp.LpAffineExpression = 0
            if x in x_to_read_quorum_vars:
                x_load += fr * sum(x_to_read_quorum_vars[x]) / read_capacity[x]
            if x in x_to_write_quorum_vars:
                x_load += ((1 - fr) * sum(x_to_write_quorum_vars[x]) /
                            write_capacity[x])
            problem += (x_load <= l, x)

        problem.solve(pulp.apis.PULP_CBC_CMD(msg=False))
        return ExplicitStrategy(nodes,
                                read_quorums,
                                [v.varValue for v in read_quorum_vars],
                                write_quorums,
                                [v.varValue for v in write_quorum_vars])
