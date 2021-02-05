from .distribution import Distribution
from .expr import choose, Expr, Node
from .quorum_system import (LATENCY, LOAD, NETWORK, NoStrategyFoundError,
                            QuorumSystem, Strategy)
from typing import Iterator, List, Optional, TypeVar
import datetime
import itertools


T = TypeVar('T')


class NoQuorumSystemFoundError(ValueError):
    pass


def _partitionings(xs: List[T]) -> Iterator[List[List[T]]]:
    """
    _partitionings(xs) yields all possible partitionings of xs. For example,
    _partitionings([1, 2, 3]) yields

        [[1], [2], [3]]
        [[1, 2], [3]]
        [[1, 3], [2]]
        [[2, 3], [1]]
        [[1, 2, 3]]
    """
    if len(xs) == 0:
        return

    def helper(xs: List[T]) -> Iterator[List[List[T]]]:
        if len(xs) == 0:
            yield []
            return

        x = xs[0]
        for partition in helper(xs[1:]):
            yield [[x]] + partition
            for i in range(len(partition)):
                yield partition[:i] + [[x] + partition[i]] + partition[i+1:]

    yield from helper(xs)


def _dup_free_exprs(nodes: List[Node[T]],
                    max_height: int = 0) -> Iterator[Expr[T]]:
    """
    _dup_free_exprs(nodes) yields all possible duplicate free expressions over
    `nodes` with height at most `max_height`. If `max_height` is not positive,
    there is no height limit. Note that an expression might be yielded more
    than once.
    """
    assert len(nodes) > 0

    if len(nodes) == 1:
        yield nodes[0]
        return

    if max_height == 1:
        for k in range(1, len(nodes) + 1):
            yield choose(k, nodes) # type: ignore
        return

    for partitioning in _partitionings(nodes):
        # We ignore the partitioning that includes every node in a single
        # partition.
        if len(partitioning) == 1:
            continue

        subiterators = [_dup_free_exprs(p, max_height-1) for p in partitioning]
        for subexprs in itertools.product(*subiterators):
            for k in range(1, len(subexprs) + 1):
                yield choose(k, list(subexprs))


def search(nodes: List[Node[T]],
           read_fraction: Optional[Distribution] = None,
           write_fraction: Optional[Distribution] = None,
           optimize: str = LOAD,
           resilience: int = 0,
           load_limit: Optional[float] = None,
           network_limit: Optional[float] = None,
           latency_limit: Optional[datetime.timedelta] = None,
           f: int = 0,
           timeout: datetime.timedelta = datetime.timedelta(seconds=0)) \
           -> QuorumSystem[T]:
    start_time = datetime.datetime.now()

    def metric(sigma: Strategy[T]) -> float:
        if optimize == LOAD:
            return sigma.load(read_fraction, write_fraction)
        elif optimize == NETWORK:
            return sigma.network_load(read_fraction, write_fraction)
        else:
            return sigma.latency(read_fraction, write_fraction).total_seconds()

    opt_qs: Optional[QuorumSystem[T]] = None
    opt_metric: Optional[float] = None

    def do_search(exprs: Iterator[Expr[T]]) -> None:
        nonlocal opt_qs
        nonlocal opt_metric

        for reads in exprs:
            qs = QuorumSystem(reads=reads)
            if qs.resilience() < resilience:
                continue

            try:
                sigma = qs.strategy(optimize = optimize,
                                    load_limit = load_limit,
                                    network_limit = network_limit,
                                    latency_limit = latency_limit,
                                    read_fraction = read_fraction,
                                    write_fraction = write_fraction,
                                    f = f)
                sigma_metric = metric(sigma)
                if opt_metric is None or sigma_metric < opt_metric:
                    opt_qs = qs
                    opt_metric = sigma_metric
            except NoStrategyFoundError:
                pass

            if (timeout != datetime.timedelta(seconds=0) and
                datetime.datetime.now() - start_time >= timeout):
                return

    do_search(_dup_free_exprs(nodes, max_height=2))
    do_search(_dup_free_exprs(nodes))

    if opt_qs is None:
        raise ValueError('no quorum system found')
    else:
        return opt_qs
