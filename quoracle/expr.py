from dataclasses import dataclass, InitVar
from typing import Any, Dict, Iterator, Generic, List, Optional, Protocol, Set, TypeVar

import datetime
import itertools
import pulp


class SupportsLessThan(Protocol):
    def __lt__(self, __other: Any) -> bool: ...


T = TypeVar('T', bound=SupportsLessThan)


def _min_hitting_set(sets: Iterator[Set]) -> int:
    x_vars: Dict[Any, pulp.LpVariable] = dict()
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


class Expr(Generic[T]):
    def __add__(self, rhs: 'Expr[T]') -> 'Expr[T]':
        def _or(lhs: Expr[T], rhs: Expr[T]) -> 'Or[T]':
            if isinstance(lhs, Or) and isinstance(rhs, Or):
                return Or(lhs.es + rhs.es)
            elif isinstance(lhs, Or):
                return Or(lhs.es + [rhs])
            elif isinstance(rhs, Or):
                return Or([lhs] + rhs.es)
            else:
                return Or([lhs, rhs])


        return _or(self, rhs)

    def __mul__(self, rhs: 'Expr[T]') -> 'Expr[T]':
        def _and(lhs: Expr[T], rhs: Expr[T]) -> 'And[T]':
            if isinstance(lhs, And) and isinstance(rhs, And):
                return And(lhs.es + rhs.es)
            elif isinstance(lhs, And):
                return And(lhs.es + [rhs])
            elif isinstance(rhs, And):
                return And([lhs] + rhs.es)
            else:
                return And([lhs, rhs])

        return _and(self, rhs)

    def quorums(self) -> Iterator[Set[T]]:
        raise NotImplementedError

    # TODO(mwhittaker): Add a function to return minimal quorums.

    # TODO(mwhittaker): Add a function to check whether two expressions are
    # equal. One simple way to do this is compare the set of minimal quorums.
    # There might be more efficient ways to check if two expressions are equal.

    def is_quorum(self, xs: Set[T]) -> bool:
        raise NotImplementedError

    def elements(self) -> Set[T]:
        return {node.x for node in self.nodes()}

    def nodes(self) -> Set['Node[T]']:
        raise NotImplementedError

    def resilience(self) -> int:
        if self.dup_free():
            return self._dup_free_min_failures() - 1
        else:
            return _min_hitting_set(self.quorums()) - 1

    def dual(self) -> 'Expr[T]':
        raise NotImplementedError

    def dup_free(self) -> bool:
        return len(self.nodes()) == self._num_leaves()

    def _num_leaves(self) -> int:
        raise NotImplementedError

    def _dup_free_min_failures(self) -> int:
        raise NotImplementedError


@dataclass
class Node(Expr[T]):
    x: T
    read_capacity: float = 1.0
    write_capacity: float = 1.0
    latency: datetime.timedelta = datetime.timedelta(seconds=1)
    capacity: InitVar[Optional[float]] = None

    def __post_init__(self, capacity):
        latency = self.latency

        # A user either specifies capacity or (read_capacity and
        # write_capacity), but not both.
        if capacity is not None and self.read_capacity != 1.0 and self.write_capacity != 1.0:
            raise ValueError('You must specify capacity or (read_capacity '
                             'and write_capacity)')

        if capacity is not None:
            self.read_capacity = capacity
            self.write_capacity = capacity

    def __hash__(self):
        return self.x.__hash__()

    def __str__(self) -> str:
        return str(self.x)

    def __repr__(self) -> str:
        return f'Node({self.x})'

    def __lt__(self, other) -> bool:
        return self.x < other.x

    def quorums(self) -> Iterator[Set[T]]:
        yield {self.x}

    def is_quorum(self, xs: Set[T]) -> bool:
        return self.x in xs

    def nodes(self) -> Set['Node[T]']:
        return {self}

    def dual(self) -> Expr:
        return self

    def _num_leaves(self) -> int:
        return 1

    def _dup_free_min_failures(self) -> int:
        return 1


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

    def _num_leaves(self) -> int:
        return sum(e._num_leaves() for e in self.es)

    def _dup_free_min_failures(self) -> int:
        return sum(e._dup_free_min_failures() for e in self.es)


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

    def _num_leaves(self) -> int:
        return sum(e._num_leaves() for e in self.es)

    def _dup_free_min_failures(self) -> int:
        return min(e._dup_free_min_failures() for e in self.es)

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

    def _num_leaves(self) -> int:
        return sum(e._num_leaves() for e in self.es)

    def _dup_free_min_failures(self) -> int:
        subfailures = [e._dup_free_min_failures() for e in self.es]
        return sum(sorted(subfailures)[:len(subfailures) - self.k + 1])


def choose(k: int, es: List[Expr[T]]) -> Expr[T]:
    if len(es) == 0:
        raise ValueError('no expressions provided')

    if not (1 <= k <= len(es)):
        raise ValueError('k must be in the range [1, len(es)]')

    if k == 1:
        return Or(es)
    elif k == len(es):
        return And(es)
    else:
        return Choose(k, es)


def majority(es: List[Expr[T]]) -> Expr[T]:
    if len(es) == 0:
        raise ValueError('no expressions provided')

    return choose(len(es) // 2 + 1, es)
