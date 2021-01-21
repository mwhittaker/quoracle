from typing import (Dict, Iterator, Generic, List, Optional, Set, Tuple,
                    TypeVar, Union)
import itertools
import numpy as np
import pulp


T = TypeVar('T')


class Expr(Generic[T]):
    def quorums(self) -> Iterator[Set[T]]:
        raise NotImplementedError

    def is_quorum(self, xs: Set[T]) -> bool:
        raise NotImplementedError

    def dual(self) -> 'Expr[T]':
        raise NotImplementedError

    def __add__(self, rhs: 'Expr[T]') -> 'Expr[T]':
        return _or(self, rhs)

    def __mul__(self, rhs: 'Expr[T]') -> 'Expr[T]':
        return _and(self, rhs)


class Node(Expr[T]):
    def __init__(self, x: T) -> None:
        self.x = x

    def __str__(self) -> str:
        return str(self.x)

    def __repr__(self) -> str:
        return f'Node({self.x})'

    def quorums(self) -> Iterator[Set[T]]:
        yield {self.x}

    def is_quorum(self, xs: Set[T]) -> bool:
        return self.x in xs

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


Distribution = Union[int, float, Dict[float, float], List[Tuple[float, float]]]


def _canonicalize_distribution(d: Distribution) -> Dict[float, float]:
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
    elif isinstance(d, list):
        return _canonicalize_distribution({f: weight for (f, weight) in d})
    else:
        raise ValueError('distribution must be an int, a float, a Dict[float, '
                         'float] or a List[Tuple[float, float]]')


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

    def strategy(self, read_fraction: Distribution) -> 'Strategy[T]':
        # TODO(mwhittaker): Implement.
        reads = list(self.read_quorums())
        writes = list(self.write_quorums())
        return ExplicitStrategy(reads, [1 / len(reads)] * len(reads),
                                writes, [1 / len(writes)] * len(writes))

    def is_read_quorum(self, xs: Set[T]) -> bool:
        return self.reads.is_quorum(xs)

    def read_quorums(self) -> Iterator[Set[T]]:
        return self.reads.quorums()

    def write_quorums(self) -> Iterator[Set[T]]:
        return self.writes.quorums()


class Strategy(Generic[T]):
    def load(self, read_fraction: Distribution) -> int:
        raise NotImplementedError

    def get_read_quorum(self) -> Set[T]:
        raise NotImplementedError

    def get_write_quorum(self) -> Set[T]:
        raise NotImplementedError


class ExplicitStrategy(Strategy[T]):
    def __init__(self,
                 reads: List[Set[T]],
                 read_weights: List[float],
                 writes: List[Set[T]],
                 write_weights: List[float]) -> None:
        self.reads = reads
        self.read_weights = read_weights
        self.writes = writes
        self.write_weights = write_weights

    # TODO(mwhittaker): Implement __str__ and __repr__.

    def load(self, read_fraction: Distribution) -> int:
        raise NotImplementedError

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
grid = QuorumSystem(reads=a*b*c + d*e*f + g*h*i)
sigma = grid.strategy(0.1)
for _ in range(10):
    print(sigma.get_write_quorum())

# - num_quorums
# - has dups?
# - optimal schedule
# - independent schedule
# - node read and write throughputs
