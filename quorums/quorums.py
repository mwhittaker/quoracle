from typing import Iterator, Generic, List, Optional, Set, TypeVar
import itertools


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


def choose(k: int, es: List[Expr[T]]) -> Choose[T]:
    return Choose(k, es)


def majority(es: List[Expr[T]]) -> Choose[T]:
    return Choose(len(es) // 2 + 1, es)


class QuorumSystem:
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


a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('g')
g = Node('g')
h = Node('h')
i = Node('i')

disjunction = a + b + c
conjunction = disjunction * disjunction * disjunction
print(conjunction)
print(conjunction.dual())
print(conjunction.dual().dual())
print(QuorumSystem(reads=conjunction))


# - num_quorums
# - has dups?
# - optimal schedule
# - independent schedule
# - node read and write throughputs
