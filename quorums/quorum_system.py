# TODO(mwhittaker): We can define a set of read quorums that are not minimal.
# Does this mess things up?

from . import distribution
from . import geometry
from .distribution import Distribution
from .expr import Expr, Node
from .geometry import Point, Segment
from typing import *
import collections
import datetime
import itertools
import numpy as np
import pulp


T = TypeVar('T')


LOAD = 'load'
NETWORK = 'network'
LATENCY = 'latency'

# TODO(mwhittaker): Add some other non-optimal strategies.
# TODO(mwhittaker): Make it easy to make arbitrary strategies.

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

    def nodes(self) -> Set[Node[T]]:
        return self.reads.nodes() | self.writes.nodes()

    def resilience(self) -> int:
        return min(self.read_resilience(), self.write_resilience())

    def read_resilience(self) -> int:
        return self.reads.resilience()

    def write_resilience(self) -> int:
        return self.writes.resilience()

    def strategy(self,
                 optimize: str = LOAD,
                 load_limit: Optional[float] = None,
                 network_limit: Optional[float] = None,
                 latency_limit: Optional[datetime.timedelta] = None,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None,
                 f: int = 0) \
                 -> 'Strategy[T]':
        if f < 0:
            raise ValueError('f must be >= 0')

        if optimize == LOAD and load_limit is not None:
            raise ValueError(
                'a load limit cannot be set when optimizing for load')

        if optimize == NETWORK and network_limit is not None:
            raise ValueError(
                'a network limit cannot be set when optimizing for network')

        if optimize == LATENCY and latency_limit is not None:
            raise ValueError(
                'a latency limit cannot be set when optimizing for latency')

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
            xs = [node.x for node in self.nodes()]
            read_quorums = list(self._f_resilient_quorums(f, xs, self.reads))
            write_quorums = list(self._f_resilient_quorums(f, xs, self.reads))
            if len(read_quorums) == 0:
                raise ValueError(f'There are no {f}-resilient read quorums')
            if len(write_quorums) == 0:
                raise ValueError(f'There are no {f}-resilient write quorums')
            return self._load_optimal_strategy(
                read_quorums,
                write_quorums,
                d,
                optimize=optimize,
                load_limit=load_limit,
                network_limit=network_limit,
                latency_limit=latency_limit)

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
        return 0
        # TODO(mwhittaker): Remove.
        # sigma = self.strategy(read_fraction, write_fraction, f)
        # return sigma.load(read_fraction, write_fraction)

    def capacity(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None,
                 f: int = 0) \
                 -> float:
        return 0
        # TODO(mwhittaker): Remove.
        # return 1 / self.load(read_fraction, write_fraction, f)

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

        We can form a linear program to compute the optimal load of this quorum
        system for some fixed read fraction fr as follows. First, we create a
        variable ri for every read quorum i and a variable wi for every write
        quorum i. ri represents the probabilty of selecting the ith read
        quorum, and wi represents the probabilty of selecting the ith write
        quorum. We introduce an additional variable l that represents the load
        and solve the following linear program.

            min L subject to
            r0 + r1 + r2 = 1
            w0 + w1 = 1
            fr (r0) + (1 - fr) (w0 + w1) <= L # a's load
            fr (r0) + (1 - fr) (w2 + w3) <= L # b's load
            fr (r1) + (1 - fr) (w0 + w2) <= L # c's load
            fr (r1) + (1 - fr) (w1 + w3) <= L # d's load

        If we assume every element x has read capacity rcap_x and write
        capacity wcap_x, then we adjust the linear program like this.

            min L subject to
            r0 + r1 + r2 = 1
            w0 + w1 = 1
            fr/rcap_a (r0) + (1 - fr)/wcap_a (w0 + w1) <= L # a's load
            fr/rcap_b (r0) + (1 - fr)/wcap_b (w2 + w3) <= L # b's load
            fr/rcap_c (r1) + (1 - fr)/wcap_c (w0 + w2) <= L # c's load
            fr/rcap_d (r1) + (1 - fr)/wcap_d (w1 + w3) <= L # d's load

        Assume we have fr = 0.9 with 80% probabilty and fr = 0.5 with 20%. Then
        we adjust the linear program as follows to find the strategy that
        minimzes the average load.

            min 0.8 * L_0.9 + 0.2 * L_0.5 subject to
            r0 + r1 + r2 = 1
            w0 + w1 = 1
            0.9/rcap_a (r0) + 0.1/wcap_a (w0 + w1) <= L_0.9 # a's load
            0.9/rcap_b (r0) + 0.1/wcap_b (w2 + w3) <= L_0.9 # b's load
            0.9/rcap_c (r1) + 0.1/wcap_c (w0 + w2) <= L_0.9 # c's load
            0.9/rcap_d (r1) + 0.1/wcap_d (w1 + w3) <= L_0.9 # d's load
            0.5/rcap_a (r0) + 0.5/wcap_a (w0 + w1) <= L_0.5 # a's load
            0.5/rcap_b (r0) + 0.5/wcap_b (w2 + w3) <= L_0.5 # b's load
            0.5/rcap_c (r1) + 0.5/wcap_c (w0 + w2) <= L_0.5 # c's load
            0.5/rcap_d (r1) + 0.5/wcap_d (w1 + w3) <= L_0.5 # d's load
        """
        nodes = self.nodes()
        x_to_node = {node.x: node for node in nodes}
        read_capacity = {node.x: node.read_capacity for node in nodes}
        write_capacity = {node.x: node.write_capacity for node in nodes}

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

        fr = sum(weight * f for (f, weight) in read_fraction.items())

        def network() -> pulp.LpAffineExpression:
            read_network = fr * sum(
                v * len(rq)
                for (rq, v) in zip(read_quorums, read_quorum_vars)
            )
            write_network = (1 - fr) * sum(
                v * len(wq)
                for (wq, v) in zip(write_quorums, write_quorum_vars)
            )
            return read_network + write_network

        def latency() -> pulp.LpAffineExpression:
            read_latency = fr * sum(
                v * self._read_quorum_latency(quorum).total_seconds()
                for (rq, v) in zip(read_quorums, read_quorum_vars)
                for quorum in [{x_to_node[x] for x in rq}]
            )
            write_latency = (1. - fr) * sum(
                v * self._write_quorum_latency(quorum).total_seconds()
                for (wq, v) in zip(write_quorums, write_quorum_vars)
                for quorum in [{x_to_node[x] for x in wq}]
            )
            return read_latency + write_latency

        def fr_load(problem: pulp.LpProblem, fr: float) -> pulp.LpAffineExpression:
            l = pulp.LpVariable(f'l_{fr}', 0, 1)

            for node in nodes:
                x = node.x
                x_load: pulp.LpAffineExpression = 0

                if x in x_to_read_quorum_vars:
                    vs = x_to_read_quorum_vars[x]
                    x_load += fr * sum(vs) / read_capacity[x]

                if x in x_to_write_quorum_vars:
                    vs = x_to_write_quorum_vars[x]
                    x_load += (1 - fr) * sum(vs) / write_capacity[x]

                problem += (x_load <= l, f'{x}{fr}')

            return l

        def load(problem: pulp.LpProblem,
                 read_fraction: Dict[float, float]) -> pulp.LpAffineExpression:
            return sum(weight * fr_load(problem, fr)
                       for (fr, weight) in read_fraction.items())

        # Form the linear program to find the load.
        problem = pulp.LpProblem("load", pulp.LpMinimize)

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
        print(problem)
        problem.solve(pulp.apis.PULP_CBC_CMD(msg=False))
        if problem.status != pulp.LpStatusOptimal:
            raise ValueError('no strategy satisfies the given constraints')

        # Prune out any quorums with 0 probability.
        non_zero_read_quorums = [
            (rq, v.varValue)
            for (rq, v) in zip(read_quorums, read_quorum_vars)
            if v.varValue != 0]
        non_zero_write_quorums = [
            (wq, v.varValue)
            for (wq, v) in zip(write_quorums, write_quorum_vars)
            if v.varValue != 0]
        return Strategy(self,
                        [rq for (rq, _) in non_zero_read_quorums],
                        [weight for (_, weight) in non_zero_read_quorums],
                        [wq for (wq, _) in non_zero_write_quorums],
                        [weight for (_, weight) in non_zero_write_quorums])


class Strategy(Generic[T]):
    def __init__(self,
                 qs: QuorumSystem[T],
                 reads: List[Set[T]],
                 read_weights: List[float],
                 writes: List[Set[T]],
                 write_weights: List[float]) -> None:
        self.qs = qs
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

    # TODO(mwhittaker): Rename throughput.
    def capacity(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None) \
                 -> float:
        return 1 / self.load(read_fraction, write_fraction)

    def network_load(self,
                     read_fraction: Optional[Distribution] = None,
                     write_fraction: Optional[Distribution] = None) -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        fr = sum(weight * f for (f, weight) in d.items())
        read_network_load = fr * sum(
            len(rq) * p
            for (rq, p) in zip(self.reads, self.read_weights)
        )
        write_network_load = (1 - fr) * sum(
            len(wq) * p
            for (wq, p) in zip(self.writes, self.write_weights)
        )
        return read_network_load + write_network_load

    def latency(self,
                read_fraction: Optional[Distribution] = None,
                write_fraction: Optional[Distribution] = None) \
                -> datetime.timedelta:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        fr = sum(weight * f for (f, weight) in d.items())

        read_latency = fr * sum((
            self.qs._read_quorum_latency(quorum) * p # type: ignore
            for (rq, p) in zip(self.reads, self.read_weights)
            for quorum in [{self.qs.x_to_node[x] for x in rq}]
        ), datetime.timedelta(seconds=0)) # type: ignore
        write_latency = (1 - fr) * sum((
            self.qs._write_quorum_latency(quorum) * p # type: ignore
            for (wq, p) in zip(self.writes, self.write_weights)
            for quorum in [{self.qs.x_to_node[x] for x in wq}]
        ), datetime.timedelta(seconds=0)) # type:ignore
        return read_latency + write_latency # type: ignore

    def node_load(self,
                  node: Node[T],
                  read_fraction: Optional[Distribution] = None,
                  write_fraction: Optional[Distribution] = None) \
                  -> float:
        d = distribution.canonicalize_rw(read_fraction, write_fraction)
        return sum(weight * self._node_load(node.x, fr)
                   for (fr, weight) in d.items())

    def node_utilization(self,
                         node: Node[T],
                         read_fraction: Optional[Distribution] = None,
                         write_fraction: Optional[Distribution] = None) \
                         -> float:
        # TODO(mwhittaker): Implement.
        return 0.0

    def node_throghput(self,
                       node: Node[T],
                       read_fraction: Optional[Distribution] = None,
                       write_fraction: Optional[Distribution] = None) \
                       -> float:
        # TODO(mwhittaker): Implement.
        return 0.0

    def _node_load(self, x: T, fr: float) -> float:
        """
        _node_load returns the load on x given a fixed read fraction fr.
        """
        fw = 1 - fr
        node = self.qs.x_to_node[x]
        return (fr * self.unweighted_read_load[x] / node.read_capacity +
                fw * self.unweighted_write_load[x] / node.write_capacity)

    def _load(self, fr: float) -> float:
        """
        _load returns the load given a fixed read fraction fr.
        """
        return max(self._node_load(node.x, fr) for node in self.qs.nodes())
