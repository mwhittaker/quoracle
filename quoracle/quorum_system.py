from . import distribution
from . import geometry
from .distribution import Distribution
from .expr import Expr, Node
from .geometry import Point, Segment
from typing import *
import collections
import datetime
import itertools
import math
import numpy as np
import pulp


T = TypeVar('T')

LOAD = 'load'
NETWORK = 'network'
LATENCY = 'latency'

# TODO(mwhittaker): Add some other non-optimal strategies.
# TODO(mwhittaker): Make it easy to make arbitrary strategies.


class NoStrategyFoundError(ValueError):
    pass


class QuorumSystem(Generic[T]):
    def __init__(self, reads: Optional[Expr[T]] = None,
                       writes: Optional[Expr[T]] = None) -> None:
        if reads is not None and writes is not None:
            optimal_writes = reads.dual()
            if not all(optimal_writes.is_quorum(wq) for wq in writes.quorums()):
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

        self.x_to_node = {node.x: node for node in self.nodes()}

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

    def node(self, x: T) -> Node[T]:
        return self.x_to_node[x]

    def nodes(self) -> Set[Node[T]]:
        return self.reads.nodes() | self.writes.nodes()

    def elements(self) -> Set[T]:
        return {node.x for node in self.nodes()}

    def resilience(self) -> int:
        return min(self.read_resilience(), self.write_resilience())

    def read_resilience(self) -> int:
        return self.reads.resilience()

    def write_resilience(self) -> int:
        return self.writes.resilience()

    def dup_free(self) -> bool:
        return self.reads.dup_free() and self.writes.dup_free()

    def load(self,
             optimize: str = LOAD,
             load_limit: Optional[float] = None,
             network_limit: Optional[float] = None,
             latency_limit: Optional[datetime.timedelta] = None,
             read_fraction: Optional[Distribution] = None,
             write_fraction: Optional[Distribution] = None,
             f: int = 0) -> float:
        return self.strategy(
            optimize=optimize,
            load_limit=load_limit,
            network_limit=network_limit,
            latency_limit=latency_limit,
            read_fraction=read_fraction,
            write_fraction=write_fraction,
            f=f
        ).load(read_fraction, write_fraction)

    def capacity(self,
                 optimize: str = LOAD,
                 load_limit: Optional[float] = None,
                 network_limit: Optional[float] = None,
                 latency_limit: Optional[datetime.timedelta] = None,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None,
                 f: int = 0) -> float:
        return self.strategy(
            optimize=optimize,
            load_limit=load_limit,
            network_limit=network_limit,
            latency_limit=latency_limit,
            read_fraction=read_fraction,
            write_fraction=write_fraction,
            f=f
        ).capacity(read_fraction, write_fraction)

    def network_load(self,
                     optimize: str = LOAD,
                     load_limit: Optional[float] = None,
                     network_limit: Optional[float] = None,
                     latency_limit: Optional[datetime.timedelta] = None,
                     read_fraction: Optional[Distribution] = None,
                     write_fraction: Optional[Distribution] = None,
                     f: int = 0) -> float:
        return self.strategy(
            optimize=optimize,
            load_limit=load_limit,
            network_limit=network_limit,
            latency_limit=latency_limit,
            read_fraction=read_fraction,
            write_fraction=write_fraction,
            f=f
        ).network_load(read_fraction, write_fraction)

    def latency(self,
                optimize: str = LOAD,
                load_limit: Optional[float] = None,
                network_limit: Optional[float] = None,
                latency_limit: Optional[datetime.timedelta] = None,
                read_fraction: Optional[Distribution] = None,
                write_fraction: Optional[Distribution] = None,
                f: int = 0) -> float:
        return self.strategy(
            optimize=optimize,
            load_limit=load_limit,
            network_limit=network_limit,
            latency_limit=latency_limit,
            read_fraction=read_fraction,
            write_fraction=write_fraction,
            f=f
        ).latency(read_fraction, write_fraction)

    def uniform_strategy(self, f: int = 0) -> 'Strategy[T]':
        """
        uniform_strategy(f) returns a uniform strategy over the minimal
        f-resilient quorums. That is, every minimal f-resilient quorum is
        equally likely to be chosen.
        """
        if f < 0:
            raise ValueError('f must be >= 0')
        elif f == 0:
            read_quorums = list(self.read_quorums())
            write_quorums = list(self.write_quorums())
        else:
            xs = list(self.elements())
            read_quorums = list(self._f_resilient_quorums(f, xs, self.reads))
            write_quorums = list(self._f_resilient_quorums(f, xs, self.reads))
            if len(read_quorums) == 0:
                raise NoStrategyFoundError(
                    f'There are no {f}-resilient read quorums')
            if len(write_quorums) == 0:
                raise NoStrategyFoundError(
                    f'There are no {f}-resilient write quorums')

        read_quorums = self._minimize(read_quorums)
        write_quorums = self._minimize(write_quorums)
        sigma_r = {frozenset(q): 1 / len(read_quorums) for q in read_quorums}
        sigma_w = {frozenset(q): 1 / len(write_quorums) for q in write_quorums}
        return Strategy(self, sigma_r, sigma_w)

    def make_strategy(self,
                      sigma_r: Dict[FrozenSet[T], float],
                      sigma_w: Dict[FrozenSet[T], float]) -> 'Strategy[T]':
        if not all(0 <= weight for weight in sigma_r.values()):
            raise ValueError('sigma_r has negative weights')
        if not all(0 <= weight for weight in sigma_w.values()):
            raise ValueError('sigma_w has negative weights')
        if not all(self.is_read_quorum(set(rq)) for rq in sigma_r):
            raise ValueError('sigma_r has non-read quorums')
        if not all(self.is_write_quorum(set(wq)) for wq in sigma_w):
            raise ValueError('sigma_w has non-write quorums')
        normalized_sigma_r = {rq: weight / sum(sigma_r.values())
                              for (rq, weight) in sigma_r.items()}
        normalized_sigma_w = {wq: weight / sum(sigma_w.values())
                              for (wq, weight) in sigma_w.items()}
        return Strategy(self,
                        sigma_r=normalized_sigma_r,
                        sigma_w=normalized_sigma_w)

    def strategy(self,
                 optimize: str = LOAD,
                 load_limit: Optional[float] = None,
                 network_limit: Optional[float] = None,
                 latency_limit: Optional[datetime.timedelta] = None,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None,
                 f: int = 0) -> 'Strategy[T]':
        if optimize not in {LOAD, NETWORK, LATENCY}:
            raise ValueError(
                f'optimize must be one of {LOAD}, {NETWORK}, or {LATENCY}')

        if optimize == LOAD and load_limit is not None:
            raise ValueError(
                'a load limit cannot be set when optimizing for load')

        if optimize == NETWORK and network_limit is not None:
            raise ValueError(
                'a network limit cannot be set when optimizing for network')

        if optimize == LATENCY and latency_limit is not None:
            raise ValueError(
                'a latency limit cannot be set when optimizing for latency')

        if f < 0:
            raise ValueError('f must be >= 0')

        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        if f == 0:
            return self._load_optimal_strategy(
                list(self.read_quorums()),
                list(self.write_quorums()),
                d,
                optimize=optimize,
                load_limit=load_limit,
                network_limit=network_limit,
                latency_limit=latency_limit)
        else:
            xs = list(self.elements())
            read_quorums = list(self._f_resilient_quorums(f, xs, self.reads))
            write_quorums = list(self._f_resilient_quorums(f, xs, self.writes))
            if len(read_quorums) == 0:
                raise NoStrategyFoundError(
                    f'There are no {f}-resilient read quorums')
            if len(write_quorums) == 0:
                raise NoStrategyFoundError(
                    f'There are no {f}-resilient write quorums')
            return self._load_optimal_strategy(
                read_quorums,
                write_quorums,
                d,
                optimize=optimize,
                load_limit=load_limit,
                network_limit=network_limit,
                latency_limit=latency_limit)

    def _minimize(self, sets: List[Set[T]]) -> List[Set[T]]:
        sets = sorted(sets, key=lambda s: len(s))
        minimal_elements: List[Set[T]] = []
        for x in sets:
            if not any(x >= y for y in minimal_elements):
                minimal_elements.append(x)
        return minimal_elements

    def _f_resilient_quorums(self,
                             f: int,
                             xs: List[T],
                             e: Expr) -> Iterator[Set[T]]:
        """
        Consider a set X of elements in xs. We say X is f-resilient if, despite
        removing an arbitrary set of f elements from X, X is a quorum in e.
        _f_resilient_quorums returns the set of all f-resilient quorums.
        """
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

    def _read_quorum_latency(self, quorum: Set[Node[T]]) -> datetime.timedelta:
        return self._quorum_latency(quorum, self.is_read_quorum)

    def _write_quorum_latency(self, quorum: Set[Node[T]]) -> datetime.timedelta:
        return self._quorum_latency(quorum, self.is_write_quorum)

    def _quorum_latency(self,
                        quorum: Set[Node[T]],
                        is_quorum: Callable[[Set[T]], bool]) \
                        -> datetime.timedelta:
        nodes = list(quorum)
        nodes.sort(key=lambda node: node.latency)
        for i in range(len(quorum)):
            if is_quorum({node.x for node in nodes[:i+1]}):
                return nodes[i].latency
        raise ValueError('_quorum_latency called on a non-quorum')

    def _load_optimal_strategy(
            self,
            read_quorums: List[Set[T]],
            write_quorums: List[Set[T]],
            read_fraction: Dict[float, float],
            optimize: str = LOAD,
            load_limit: Optional[float] = None,
            network_limit: Optional[float] = None,
            latency_limit: Optional[datetime.timedelta] = None) -> 'Strategy[T]':
        """
        Consider the following 2x2 grid quorum system.

            a   b

            c   d

        with

            read_quorums = [{a, b}, {c, d}]
            write_quorums = [{a, c}, {a, d}, {b, c}, {b, d}]

        We want to find the strategy that is optimal with respect to load,
        network load, or latency that satisfies the provided load, network
        load, or latency constraints.

        We can find the optimal strategy using linear programming. First, we
        create a variable ri for every read quorum i and a variable wi for
        every write quorum i. ri represents the probabilty of selecting the ith
        read quorum, and wi represents the probabilty of selecting the ith
        write quorum.

        We now explain how to represent load, network load, and latency as
        linear expressions.

        Load
        ====
        Assume a read fraction fr and write fraction fw. The load of a node a is

            load(a) = (fr * rprob(a) / rcap(a)) + (fw * wprob(a) / wcap(a))

        where prob_r(a) and prob_w(a) are the probabilities that a is selected
        as part of a read or write quorum respectively; and rcap(a) and wcap(a)
        are the read and write capacities of a. We can express prob_r(a) and
        prob_w(a) as follows:

                  rprob(a) = sum({ri | a is in read quorum i})
                  wprob(a) = sum({wi | a is in write quorum i})

        Using the example grid quorum above, we have:

                      rprob(a) = r0    wprob(a) = w0 + w1
                      rprob(b) = r0    wprob(b) = w2 + w3
                      rprob(c) = r1    wprob(c) = w0 + w2
                      rprob(d) = r1    wprob(d) = w1 + w3

        The load of a strategy is the maximum load on any node. We can compute
        this by minimizing a new variable l and constraining the load of every
        node to be less than l. Using the example above, we have

               min l subject to
               fr * r0 * rcap(a) + fw * (w0 + w1) * wcap(a) <= l
               fr * r0 * rcap(b) + fw * (w2 + w3) * wcap(b) <= l
               fr * r1 * rcap(c) + fw * (w0 + w2) * wcap(c) <= l
               fr * r1 * rcap(d) + fw * (w1 + w3) * wcap(d) <= l

        To compute the load of a strategy with respect to a distribution of
        read_fractions, we compute the load for every value of fr and weight
        according to the distribution. For example, imagine fr is 0.9 80% of
        the time and 0.5 20% of the time. We have:

               min 0.8 * l0.9 + 0.2 * l0.5
               0.9 * r0 * rcap(a) + 0.1 * (w0 + w1) * wcap(a) <= l0.9
               0.9 * r0 * rcap(b) + 0.1 * (w2 + w3) * wcap(b) <= l0.9
               0.9 * r1 * rcap(c) + 0.1 * (w0 + w2) * wcap(c) <= l0.9
               0.9 * r1 * rcap(d) + 0.1 * (w1 + w3) * wcap(d) <= l0.9
               0.5 * r0 * rcap(a) + 0.5 * (w0 + w1) * wcap(a) <= l0.5
               0.5 * r0 * rcap(b) + 0.5 * (w2 + w3) * wcap(b) <= l0.5
               0.5 * r1 * rcap(c) + 0.5 * (w0 + w2) * wcap(c) <= l0.5
               0.5 * r1 * rcap(d) + 0.5 * (w1 + w3) * wcap(d) <= l0.5

        Let the expression for load be LOAD.

        Network
        =======
        The network load of a strategy is the expected size of a quorum. For a
        fixed fr, We can compute the network load as:

                     fr * sum_i(size(read quorum i) * ri) +
                     fw * sum_i(size(write quorum i) * ri)

        Using the example above:

               fr * (2*r0 + 2*r1) + fw * (2*w0 + 2*w1 + 2*w2 + 2*w3)

        For a distribution of read fractions, we compute the weighted average.
        Let the expression for network load be NETWORK.

        Latency
        =======
        The latency of a strategy is the expected latency of a quorum. We can
        compute the latency as:

                     fr * sum_i(latency(read quorum i) * ri) +
                     fw * sum_i(latency(write quorum i) * ri)

        Using the example above (assuming every node has a latency of 1):

               fr * (1*r0 + 1*r1) + fw * (1*w0 + 1*w1 + 1*w2 + 1*w3)

        For a distribution of read fractions, we compute the weighted average.
        Let the expression for latency be LATENCY.

        Linear Program
        ==============
        To find an optimal strategy, we use a linear program. The objective
        specified by the user is minimized, and any provided constraints are
        added as constraints to the program. For example, imagine the user
        wants a load optimal strategy with network load <= 2 and latency <= 3.
        We form the program:

            min LOAD subject to
            sum_i(ri) = 1 # ensure we have a valid distribution on read quorums
            sum_i(wi) = 1 # ensure we have a valid distribution on write quorums
            NETWORK <= 2
            LATENCY <= 3

        Using the example above assuming a fixed fr, we have:

               min l subject to
               fr * r0 * rcap(a) + fw * (w0 + w1) * wcap(a) <= l
               fr * r0 * rcap(b) + fw * (w2 + w3) * wcap(b) <= l
               fr * r1 * rcap(c) + fw * (w0 + w2) * wcap(c) <= l
               fr * r1 * rcap(d) + fw * (w1 + w3) * wcap(d) <= l
               fr * (2*r0 + 2*r1) + fw * (2*w0 + 2*w1 + 2*w2 + 2*w3) <= 2
               fr * (1*r0 + 1*r1) + fw * (1*w0 + 1*w1 + 1*w2 + 1*w3) <= 3

        If we instead wanted to minimize network load with load <= 4 and
        latency <= 5, we would have the following program:

               min fr * (2*r0 + 2*r1) +
                   fw * (2*w0 + 2*w1 + 2*w2 + 2*w3) subject to
               fr * r0 * rcap(a) + fw * (w0 + w1) * wcap(a) <= 4
               fr * r0 * rcap(b) + fw * (w2 + w3) * wcap(b) <= 4
               fr * r1 * rcap(c) + fw * (w0 + w2) * wcap(c) <= 4
               fr * r1 * rcap(d) + fw * (w1 + w3) * wcap(d) <= 4
               fr * (1*r0 + 1*r1) + fw * (1*w0 + 1*w1 + 1*w2 + 1*w3) <= 5
        """
        # Create a variable for every read quorum and every write quorum. While
        # we do this, map each element x to the read and write quorums that
        # it's in. For example, image we have the following read and write
        # quorums:
        #
        #     read_quorums = [{a}, {a, b}, {a, c}]
        #     write_quorums = [{a, b}, {a, b, c}]
        #
        # Then, we'd have
        #
        #     read_quorum_vars = [r0, r1, 2]
        #     write_quorum_vars = [w0, w1]
        #     x_to_read_quorum_vars = {a: [r1, r2, r3], b: [r1], c: [r2]}
        #     x_to_write_quorum_vars = {a: [w1, w2], b: [w2, w2], c: [w2]}
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

        fr = sum(p * fr for (fr, p) in read_fraction.items())

        def network() -> pulp.LpAffineExpression:
            reads = fr * sum(
                v * len(rq)
                for (rq, v) in zip(read_quorums, read_quorum_vars)
            )
            writes = (1 - fr) * sum(
                v * len(wq)
                for (wq, v) in zip(write_quorums, write_quorum_vars)
            )
            return reads + writes

        def latency() -> pulp.LpAffineExpression:
            reads = fr * sum(
                v * self._read_quorum_latency(quorum).total_seconds()
                for (rq, v) in zip(read_quorums, read_quorum_vars)
                for quorum in [{self.node(x) for x in rq}]
            )
            writes = (1 - fr) * sum(
                v * self._write_quorum_latency(quorum).total_seconds()
                for (wq, v) in zip(write_quorums, write_quorum_vars)
                for quorum in [{self.node(x) for x in wq}]
            )
            return reads + writes

        def fr_load(problem: pulp.LpProblem, fr: float) -> pulp.LpAffineExpression:
            l = pulp.LpVariable(f'l_{fr}', 0, 1)

            for node in self.nodes():
                x = node.x
                x_load: pulp.LpAffineExpression = 0

                if x in x_to_read_quorum_vars:
                    vs = x_to_read_quorum_vars[x]
                    x_load += fr * sum(vs) / self.node(x).read_capacity

                if x in x_to_write_quorum_vars:
                    vs = x_to_write_quorum_vars[x]
                    x_load += (1 - fr) * sum(vs) / self.node(x).write_capacity

                problem += (x_load <= l, f'{x}{fr}')

            return l

        def load(problem: pulp.LpProblem,
                 read_fraction: Dict[float, float]) -> pulp.LpAffineExpression:
            return sum(p * fr_load(problem, fr)
                       for (fr, p) in read_fraction.items())

        # Form the linear program.
        problem = pulp.LpProblem("optimal_strategy", pulp.LpMinimize)

        # We add these constraints to make sure that the probabilities we
        # select form valid probabilty distributions.
        problem += (sum(read_quorum_vars) == 1, 'valid read strategy')
        problem += (sum(write_quorum_vars) == 1, 'valid write strategy')

        # Add the objective.
        if optimize == LOAD:
            problem += load(problem, read_fraction)
        elif optimize == NETWORK:
            problem += network()
        else:
            assert optimize == LATENCY
            problem += latency()

        # Add any constraints.
        if load_limit is not None:
            problem += (load(problem, read_fraction) <= load_limit,
                        'load limit')

        if network_limit is not None:
            problem += (network() <= network_limit, 'network limit')

        if latency_limit is not None:
            problem += (latency() <= latency_limit.total_seconds(),
                        'latency limit')

        # Solve the linear program.
        problem.solve(pulp.apis.PULP_CBC_CMD(msg=False))
        if problem.status != pulp.LpStatusOptimal:
            raise NoStrategyFoundError(
                'no strategy satisfies the given constraints')

        # Prune out any quorums with 0 probability.
        sigma_r = {
            frozenset(rq): v.varValue
            for (rq, v) in zip(read_quorums, read_quorum_vars)
            if v.varValue != 0
        }
        sigma_w = {
            frozenset(wq): v.varValue
            for (wq, v) in zip(write_quorums, write_quorum_vars)
            if v.varValue != 0
        }

        return Strategy(self, sigma_r, sigma_w)


class Strategy(Generic[T]):
    def __init__(self,
                 qs: QuorumSystem[T],
                 sigma_r: Dict[FrozenSet[T], float],
                 sigma_w: Dict[FrozenSet[T], float]) -> None:
        self.qs = qs
        self.sigma_r = sigma_r
        self.sigma_w = sigma_w

        # The probability that x is chosen as part of a read quorum.
        self.x_read_probability: Dict[T, float] = collections.defaultdict(float)
        for (read_quorum, p) in self.sigma_r.items():
            for x in read_quorum:
                self.x_read_probability[x] += p

        # The probability that x is chosen as part of a write quorum.
        self.x_write_probability: Dict[T, float] = collections.defaultdict(float)
        for (write_quorum, weight) in self.sigma_w.items():
            for x in write_quorum:
                self.x_write_probability[x] += weight

    @no_type_check
    def __str__(self) -> str:
        # T may not comparable, so mypy complains about this sort.
        reads = {tuple(sorted(rq)): p for (rq, p) in self.sigma_r.items()}
        writes = {tuple(sorted(wq)): p for (wq, p) in self.sigma_w.items()}
        return f'Strategy(reads={reads}, writes={writes})'

    def quorum_system(self) -> QuorumSystem[T]:
        return self.qs

    def node(self, x: T) -> Node[T]:
        return self.qs.node(x)

    def nodes(self) -> Set[Node[T]]:
        return self.qs.nodes()

    def get_read_quorum(self) -> Set[T]:
        return set(np.random.choice(list(self.sigma_r.keys()),
                                    p=list(self.sigma_r.values())))

    def get_write_quorum(self) -> Set[T]:
        return set(np.random.choice(list(self.sigma_w.keys()),
                                    p=list(self.sigma_w.values())))

    def load(self,
             read_fraction: Optional[Distribution] = None,
             write_fraction: Optional[Distribution] = None) -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(p * self._load(fr) for (fr, p) in d.items())

    def capacity(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None) -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(p * 1 / self._load(fr) for (fr, p) in d.items())

    def network_load(self,
                     read_fraction: Optional[Distribution] = None,
                     write_fraction: Optional[Distribution] = None) -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        fr = sum(p * fr for (fr, p) in d.items())
        reads = fr * sum(p * len(rq) for (rq, p) in self.sigma_r.items())
        writes = (1 - fr) * sum(p * len(wq) for (wq, p) in self.sigma_w.items())
        return reads + writes

    # mypy doesn't like calling sum with timedeltas.
    @no_type_check
    def latency(self,
                read_fraction: Optional[Distribution] = None,
                write_fraction: Optional[Distribution] = None) \
                -> datetime.timedelta:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        fr = sum(p * fr for (fr, p) in d.items())

        reads = fr * sum((
            p * self.qs._read_quorum_latency({self.node(x) for x in rq})
            for (rq, p) in self.sigma_r.items()
        ), datetime.timedelta(seconds=0))

        writes = (1 - fr) * sum((
            p * self.qs._write_quorum_latency({self.node(x) for x in wq})
            for (wq, p) in self.sigma_w.items()
        ), datetime.timedelta(seconds=0))

        return reads + writes

    def node_load(self,
                  node: Node[T],
                  read_fraction: Optional[Distribution] = None,
                  write_fraction: Optional[Distribution] = None) -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(p * self._node_load(node, fr) for (fr, p) in d.items())

    def node_utilization(self,
                         node: Node[T],
                         read_fraction: Optional[Distribution] = None,
                         write_fraction: Optional[Distribution] = None) \
                         -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(p * self._node_utilization(node, fr)
                   for (fr, p) in d.items())

    def node_throughput(self,
                        node: Node[T],
                        read_fraction: Optional[Distribution] = None,
                        write_fraction: Optional[Distribution] = None) -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(p * self._node_throughput(node, fr) for (fr, p) in d.items())

    def _load(self, fr: float) -> float:
        return max(self._node_load(node, fr) for node in self.nodes())

    def _node_load(self, node: Node[T], fr: float) -> float:
        fw = 1 - fr
        return (fr * self.x_read_probability[node.x] / node.read_capacity +
                fw * self.x_write_probability[node.x] / node.write_capacity)

    def _node_utilization(self, node: Node[T], fr: float) -> float:
        return self._node_load(node, fr) / self._load(fr)

    def _node_throughput(self, node: Node[T], fr: float) -> float:
        cap = 1 / self._load(fr)
        fw = 1 - fr
        return cap * (fr * self.x_read_probability[node.x] +
                      fw * self.x_write_probability[node.x])
