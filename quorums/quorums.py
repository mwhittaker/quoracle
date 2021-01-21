from typing import Iterator, Generic, List, Set, TypeVar
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


def _and(lhs: Expr[T], rhs: Expr[T]) -> Expr[T]:
    if isinstance(lhs, And) and isinstance(rhs, And):
        return And(lhs.es + rhs.es)
    elif isinstance(lhs, And):
        return And(lhs.es + [rhs])
    elif isinstance(rhs, And):
        return And([lhs] + rhs.es)
    else:
        return And([lhs, rhs])


def _or(lhs: Expr[T], rhs: Expr[T]) -> Expr[T]:
    if isinstance(lhs, Or) and isinstance(rhs, Or):
        return Or(lhs.es + rhs.es)
    elif isinstance(lhs, Or):
        return Or(lhs.es + [rhs])
    elif isinstance(rhs, Or):
        return Or([lhs] + rhs.es)
    else:
        return Or([lhs, rhs])


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


class QuorumSystem:
    def __init__(self, reads, writes) -> None:
        pass


a = Node('a')
b = Node('b')
c = Node('c')
disjunction = a + b + c
conjunction = disjunction * disjunction * disjunction
print(conjunction)
print(conjunction.dual())
print(conjunction.dual().dual())
