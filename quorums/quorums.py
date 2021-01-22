# TODO(mwhittaker): We can define a set of read quorums that are not minimal.
# Does this mess things up?

from typing import (Dict, Iterator, Generic, List, Optional, Set, Tuple,
                    TypeVar, Union)
import collections
import itertools
import numpy as np
import pulp


T = TypeVar('T')


class Expr(Generic[T]):
    def __add__(self, rhs: 'Expr[T]') -> 'Expr[T]':
        return _or(self, rhs)

    def __mul__(self, rhs: 'Expr[T]') -> 'Expr[T]':
        return _and(self, rhs)

    def quorums(self) -> Iterator[Set[T]]:
        raise NotImplementedError

    def is_quorum(self, xs: Set[T]) -> bool:
        raise NotImplementedError

    def elements(self) -> Set[T]:
        return {node.x for node in self.nodes()}

    def nodes(self) -> Set['Node[T]']:
        raise NotImplementedError

    def dual(self) -> 'Expr[T]':
        raise NotImplementedError


class Node(Expr[T]):
    def __init__(self,
                 x: T,
                 capacity: Optional[float] = None,
                 read_capacity: Optional[float] = None,
                 write_capacity: Optional[float] = None) -> None:
        self.x = x

        # A user either specifies capacity or (read_capacity and
        # write_capacity), but not both.
        if (capacity is None and
            read_capacity is None and
            write_capacity is None):
            self.read_capacity = 1.0
            self.write_capacity = 1.0
        elif (capacity is not None and
              read_capacity is None and
              write_capacity is None):
            self.read_capacity = capacity
            self.write_capacity = capacity
        elif (capacity is None and
              read_capacity is not None and
              write_capacity is not None):
            self.read_capacity = read_capacity
            self.write_capacity = write_capacity
        else:
            raise ValueError('You must specify capacity or (read_capacity '
                             'and write_capacity)')

    def __str__(self) -> str:
        return str(self.x)

    def __repr__(self) -> str:
        return f'Node({self.x})'

    def quorums(self) -> Iterator[Set[T]]:
        yield {self.x}

    def is_quorum(self, xs: Set[T]) -> bool:
        return self.x in xs

    def nodes(self) -> Set['Node[T]']:
        return {self}

    def dual(self) -> Expr:
        return self


class Or(Expr[T]):
    def __init__(self, es: List[Expr[T]]) -> None:
        if len(es) == 0:
            raise ValueError(f'Or cannot be constructed with an empty list')

        self.es = es

    def __str__(self) -> str:
        return '(' + ' + '.join(str(e) for e in self.es) + ')'

    def __repr__(self) -> str:
        return f'Or({self.es})'

    def quorums(self) -> Iterator[Set[T]]:
        for e in self.es:
            yield from e.quorums()

    def is_quorum(self, xs: Set[T]) -> bool:
        return any(e.is_quorum(xs) for e in self.es)

    def nodes(self) -> Set[Node[T]]:
        return set.union(*[e.nodes() for e in self.es])

    def dual(self) -> Expr:
        return And([e.dual() for e in self.es])


class And(Expr[T]):
    def __init__(self, es: List[Expr[T]]) -> None:
        if len(es) == 0:
            raise ValueError(f'And cannot be constructed with an empty list')

        self.es = es

    def __str__(self) -> str:
        return '(' + ' * '.join(str(e) for e in self.es) + ')'

    def __repr__(self) -> str:
        return f'And({self.es})'

    def quorums(self) -> Iterator[Set[T]]:
        for subquorums in itertools.product(*[e.quorums() for e in self.es]):
            yield set.union(*subquorums)

    def is_quorum(self, xs: Set[T]) -> bool:
        return all(e.is_quorum(xs) for e in self.es)

    def nodes(self) -> Set[Node[T]]:
        return set.union(*[e.nodes() for e in self.es])

    def dual(self) -> Expr:
        return Or([e.dual() for e in self.es])


class Choose(Expr[T]):
    def __init__(self, k: int, es: List[Expr[T]]) -> None:
        if k <= 0 or k > len(es):
            raise ValueError(f'k must be in the range [1, {len(es)}]')

        self.k = k
        self.es = es

    def __str__(self) -> str:
        return f'choose{self.k}(' + ', '.join(str(e) for e in self.es) + ')'

    def __repr__(self) -> str:
        return f'Chose({self.k}, {self.es})'

    def quorums(self) -> Iterator[Set[T]]:
        for combo in itertools.combinations(self.es, self.k):
            for subquorums in itertools.product(*[e.quorums() for e in combo]):
                yield set.union(*subquorums)

    def is_quorum(self, xs: Set[T]) -> bool:
        return sum(1 if e.is_quorum(xs) else 0 for e in self.es) >= self.k

    def nodes(self) -> Set[Node[T]]:
        return set.union(*[e.nodes() for e in self.es])

    def dual(self) -> Expr:
        # TODO(mwhittaker): Prove that this is in fact the dual.
        return Choose(len(self.es) - self.k + 1, [e.dual() for e in self.es])


def _and(lhs: Expr[T], rhs: Expr[T]) -> 'And[T]':
    if isinstance(lhs, And) and isinstance(rhs, And):
        return And(lhs.es + rhs.es)
    elif isinstance(lhs, And):
        return And(lhs.es + [rhs])
    elif isinstance(rhs, And):
        return And([lhs] + rhs.es)
    else:
        return And([lhs, rhs])


def _or(lhs: Expr[T], rhs: Expr[T]) -> 'Or[T]':
    if isinstance(lhs, Or) and isinstance(rhs, Or):
        return Or(lhs.es + rhs.es)
    elif isinstance(lhs, Or):
        return Or(lhs.es + [rhs])
    elif isinstance(rhs, Or):
        return Or([lhs] + rhs.es)
    else:
        return Or([lhs, rhs])


def choose(k: int, es: List[Expr[T]]) -> Expr[T]:
    if k == 1:
        return Or(es)
    elif k == len(es):
        return And(es)
    else:
        return Choose(k, es)


def majority(es: List[Expr[T]]) -> Expr[T]:
    return choose(len(es) // 2 + 1, es)


ReadFraction = float
ReadWriteFraction = float
Weight = float
Probability = float
Distribution = Union[
    # For example, 1 means 100% reads.
    int,
    # For example, 0.25 means 25% reads.
    float,
    # For example, {0.25: 1, 0.8: 2} means 25% reads one third of the time and
    # 80% reads two thirds of the time.
    Dict[ReadWriteFraction, Weight],
]


def _canonicalize_distribution(d: Distribution) \
        -> Dict[ReadWriteFraction, Probability]:
    if isinstance(d, int):
        if d < 0 or d > 1:
            raise ValueError('distribution must be in the range [0, 1]')
        return {float(d): 1.}
    elif isinstance(d, float):
        if d < 0 or d > 1:
            raise ValueError('distribution must be in the range [0, 1]')
        return {d: 1.}
    elif isinstance(d, dict):
        if len(d) == 0:
            raise ValueError('distribution cannot empty')

        if any(weight < 0 for weight in d.values()):
            raise ValueError('distribution cannot have negative weights')

        total_weight = sum(d.values())
        if total_weight == 0:
            raise ValueError('distribution cannot have zero weight')

        return {float(f): weight / total_weight
                for (f, weight) in d.items()
                if weight > 0}
    else:
        raise ValueError('distribution must be an int, a float, a Dict[float, '
                         'float] or a List[Tuple[float, float]]')


def _canonicalize_rw_distribution(read_fraction: Optional[Distribution],
                                  write_fraction: Optional[Distribution]) \
                                  -> Dict[ReadFraction, Probability]:
    if read_fraction is None and write_fraction is None:
        raise ValueError('Either read_fraction or write_fraction must be given')
    elif read_fraction is not None and write_fraction is not None:
        raise ValueError('Only one of read_fraction or write_fraction can be '
                         'given')
    elif read_fraction is not None:
        return _canonicalize_distribution(read_fraction)
    else:
        assert write_fraction is not None
        return {1 - f: weight
                for (f, weight) in
                _canonicalize_distribution(write_fraction).items()}


class QuorumSystem(Generic[T]):
    def __init__(self, reads: Optional[Expr[T]] = None,
                       writes: Optional[Expr[T]] = None) -> None:
        if reads is not None and writes is not None:
            # TODO(mwhittaker): Think of ways to make this more efficient.
            assert all(len(r & w) > 0
                       for (r, w) in itertools.product(reads.quorums(),
                                                       writes.quorums()))
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
        return self._min_hitting_set(self.read_quorums()) - 1

    def write_resilience(self) -> int:
        return self._min_hitting_set(self.write_quorums()) - 1

    def strategy(self,
                 read_fraction: Optional[Distribution] = None,
                 write_fraction: Optional[Distribution] = None,
                 f: int = 0) \
                 -> 'Strategy[T]':
        if f < 0:
            raise ValueError('f must be >= 0')

        d = _canonicalize_rw_distribution(read_fraction, write_fraction)
        if f == 0:
            return self._load_optimal_strategy(
                list(self.read_quorums()),
                list(self.write_quorums()),
                d)
        else:
            xs = [node.x for node in self.nodes()]
            return self._load_optimal_strategy(
                list(self._f_resilient_quorums(f, xs, self.reads)),
                list(self._f_resilient_quorums(f, xs, self.writes)),
                d)

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

    def _min_hitting_set(self, sets: Iterator[Set[T]]) -> int:
        x_vars: Dict[T, pulp.LpVariable] = dict()
        next_id = itertools.count()

        problem = pulp.LpProblem("min_hitting_set", pulp.LpMinimize)
        for (i, xs) in enumerate(sets):
            for x in xs:
                if x not in x_vars:
                    id = next(next_id)
                    x_vars[x] = pulp.LpVariable(f'x{id}', cat=pulp.LpBinary)
            problem += sum(x_vars[x] for x in xs) >= 1

        problem += sum(x_vars.values())
        problem.solve(pulp.apis.PULP_CBC_CMD(msg=False))
        return int(sum(v.varValue for v in x_vars.values()))

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

        # print(problem)
        problem.solve(pulp.apis.PULP_CBC_CMD(msg=False))
        return ExplicitStrategy(nodes,
                                read_quorums,
                                [v.varValue for v in read_quorum_vars],
                                write_quorums,
                                [v.varValue for v in write_quorum_vars])
        # for v in read_weights + write_weights:
        #     print(f'{v.name} = {v.varValue}')
        # return l.varValue



class Strategy(Generic[T]):
    def load(self,
             read_fraction: Optional[Distribution] = None,
             write_fraction: Optional[Distribution] = None) \
             -> float:
        raise NotImplementedError

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
        d = _canonicalize_rw_distribution(read_fraction, write_fraction)
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




a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')
g = Node('g')
h = Node('h')
i = Node('i')

walls = QuorumSystem(reads=a*b + c*d*e)
paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)
maj = QuorumSystem(reads=majority([a, b, c, d, e]))

for qs in [walls, paths, maj]:
    sigma_0 = qs.strategy(read_fraction=0.5)
    sigma_1 = qs.strategy(read_fraction=0.5, f=1)
    print(sigma_0.load(read_fraction=0.5), sigma_1.load(read_fraction=0.5))
    print(sigma_1)

#
# qs = QuorumSystem(reads = a*b + a*c)
# print(list(qs.read_quorums()))
# sigma = qs.strategy(read_fraction=0.5)
# print(list(qs.write_quorums()))
# print(sigma)
# print(1 / sigma.load(read_fraction=0.5))

# paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)
# print(paths.resilience())
# sigma = paths.strategy(read_fraction=0.5)
# print(sigma.load(read_fraction=0.5))
#
# walls = QuorumSystem(reads=a*b + c*d*e)
# print(walls.resilience())
# sigma = walls.strategy(read_fraction=0.5)
# print(sigma.load(read_fraction=0.5))



# wpaxos = QuorumSystem(reads=majority([majority([a, b, c]),
#                                       majority([d, e, f]),
#                                       majority([g, h, i])]))
# sigma_1 = wpaxos.strategy(read_fraction=0.1)
# sigma_5 = wpaxos.strategy(read_fraction=0.5)
# sigma_9 = wpaxos.strategy(read_fraction=0.9)
# sigma_even = wpaxos.strategy(read_fraction={0.1: 2, 0.5: 2, 0.9: 1})
# for sigma in [sigma_1, sigma_5, sigma_9, sigma_even]:
#     frs = [0.1, 0.5, 0.9, {0.1: 2, 0.5: 2, 0.9: 1}]
#     print([sigma.load(fr) for fr in frs])

# - num_quorums
# - has dups?
# - optimal schedule
# - independent schedule
# - node read and write throughputs
