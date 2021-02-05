from quoracle import *
from quoracle.expr import Expr
from quoracle.search import _dup_free_exprs, _partitionings
from typing import Any, FrozenSet, List
import datetime
import unittest


class TestSearch(unittest.TestCase):
    def test_partitions(self):
        def setify(partitions: List[List[List[Any]]]) \
                   -> FrozenSet[FrozenSet[FrozenSet[Any]]]:
            return frozenset(frozenset(frozenset(s) for s in partition)
                             for partition in partitions)

        def assert_equal(x: List[List[Any]], y: List[List[Any]]):
            self.assertEqual(setify(x), setify(y))

        assert_equal(list(_partitionings([])), [])
        assert_equal(list(_partitionings([1])), [[[1]]])
        assert_equal(list(_partitionings([1, 2])), [
            [[1], [2]],
            [[1, 2]],
        ])
        assert_equal(list(_partitionings([1, 2, 3])), [
            [[1], [2], [3]],
            [[1, 2], [3]],
            [[1, 3], [2]],
            [[2, 3], [1]],
            [[1, 2, 3]],
        ])
        assert_equal(list(_partitionings([1, 2, 3, 4])), [
            [[1], [2], [3], [4]],
            [[1, 2], [3], [4]],
            [[1, 3], [2], [4]],
            [[1, 4], [2], [3]],
            [[2, 3], [1], [4]],
            [[2, 4], [1], [3]],
            [[3, 4], [1], [2]],
            [[1, 2], [3, 4]],
            [[1, 3], [2, 4]],
            [[1, 4], [2, 3]],
            [[2, 3, 4], [1]],
            [[1, 3, 4], [2]],
            [[1, 2, 4], [3]],
            [[1, 2, 3], [4]],
            [[1, 2, 3, 4]],
        ])

    def test_dup_free_exprs(self):
        def quorums(e: Expr) -> FrozenSet[FrozenSet[Any]]:
            return frozenset(frozenset(q) for q in e.quorums())

        def assert_equal(xs: List[Expr], ys: List[Expr]) -> None:
            self.assertEqual(frozenset(quorums(x) for x in xs),
                             frozenset(quorums(y) for y in ys))

        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')

        assert_equal(list(_dup_free_exprs([a])), [a])
        assert_equal(list(_dup_free_exprs([a, b])), [
            a + b,
            a * b,
        ])
        assert_equal(list(_dup_free_exprs([a, b, c])), [
            a + b + c,
            choose(2, [a, b, c]),
            a * b * c,
            (a + b) + c,
            (a + b) * c,
            (a * b) + c,
            (a * b) * c,
            (a + c) + b,
            (a + c) * b,
            (a * c) + b,
            (a * c) * b,
            (b + c) + a,
            (b + c) * a,
            (b * c) + a,
            (b * c) * a,
        ])
        assert_equal(list(_dup_free_exprs([a, b, c], max_height=1)), [
            a + b + c,
            choose(2, [a, b, c]),
            a * b * c,
        ])
        assert_equal(list(_dup_free_exprs([a, b, c, d], max_height=1)), [
            a + b + c + d,
            choose(2, [a, b, c, d]),
            choose(3, [a, b, c, d]),
            a * b * c * d,
        ])
        assert_equal(list(_dup_free_exprs([a, b, c, d], max_height=2)), [
            a + b + c + d,
            choose(2, [a, b, c, d]),
            choose(3, [a, b, c, d]),
            a * b * c * d,

            (a + b) + c + d,
            (a + b) * c * d,
            (a * b) + c + d,
            (a * b) * c * d,
            choose(2, [a + b, c, d]),
            choose(2, [a * b, c, d]),

            (a + c) + b + d,
            (a + c) * b * d,
            (a * c) + b + d,
            (a * c) * b * d,
            choose(2, [a + c, b, d]),
            choose(2, [a * c, b, d]),

            (a + d) + b + c,
            (a + d) * b * c,
            (a * d) + b + c,
            (a * d) * b * c,
            choose(2, [a + d, b, c]),
            choose(2, [a * d, b, c]),

            (b + c) + a + d,
            (b + c) * a * d,
            (b * c) + a + d,
            (b * c) * a * d,
            choose(2, [b + c, a, d]),
            choose(2, [b * c, a, d]),

            (b + d) + a + c,
            (b + d) * a * c,
            (b * d) + a + c,
            (b * d) * a * c,
            choose(2, [b + d, a, c]),
            choose(2, [b * d, a, c]),

            (c + d) + a + b,
            (c + d) * a * b,
            (c * d) + a + b,
            (c * d) * a * b,
            choose(2, [c + d, a, b]),
            choose(2, [c * d, a, b]),

            (a + b) + (c + d),
            (a + b) + (c * d),
            (a + b) * (c + d),
            (a + b) * (c * d),
            (a * b) + (c + d),
            (a * b) + (c * d),
            (a * b) * (c + d),
            (a * b) * (c * d),

            (a + c) + (b + d),
            (a + c) + (b * d),
            (a + c) * (b + d),
            (a + c) * (b * d),
            (a * c) + (b + d),
            (a * c) + (b * d),
            (a * c) * (b + d),
            (a * c) * (b * d),

            (a + d) + (b + c),
            (a + d) + (b * c),
            (a + d) * (b + c),
            (a + d) * (b * c),
            (a * d) + (b + c),
            (a * d) + (b * c),
            (a * d) * (b + c),
            (a * d) * (b * c),

            a + (b + c + d),
            a + (b * c * d),
            a * (b + c + d),
            a * (b * c * d),
            a + choose(2, [b, c, d]),
            a * choose(2, [b, c, d]),

            b + (a + c + d),
            b + (a * c * d),
            b * (a + c + d),
            b * (a * c * d),
            b + choose(2, [a, c, d]),
            b * choose(2, [a, c, d]),

            c + (a + b + d),
            c + (a * b * d),
            c * (a + b + d),
            c * (a * b * d),
            c + choose(2, [a, b, d]),
            c * choose(2, [a, b, d]),

            d + (a + b + c),
            d + (a * b * c),
            d * (a + b + c),
            d * (a * b * c),
            d + choose(2, [a, b, c]),
            d * choose(2, [a, b, c]),
        ])

    def test_search(self):
        a = Node('a', capacity=1, latency=datetime.timedelta(seconds=2))
        b = Node('b', capacity=2, latency=datetime.timedelta(seconds=1))
        c = Node('c', capacity=1, latency=datetime.timedelta(seconds=2))
        d = Node('d', capacity=2, latency=datetime.timedelta(seconds=1))
        e = Node('e', capacity=1, latency=datetime.timedelta(seconds=2))
        f = Node('f', capacity=2, latency=datetime.timedelta(seconds=1))

        for fr in [0, 0.5, 1]:
            search([a, b, c], read_fraction=fr)
            search([a, b, c], read_fraction=fr, optimize='network')
            search([a, b, c], read_fraction=fr, optimize='latency')
            search([a, b, c], read_fraction=fr, resilience=1)
            search([a, b, c], read_fraction=fr, f=1)

        search([a, b, c],
               read_fraction=0.25,
               network_limit=3,
               latency_limit=datetime.timedelta(seconds=2))

        for fr in [0, 0.5]:
            t = datetime.timedelta(seconds=0.25)
            nodes = [a, b, c, d, e, f]
            search(nodes, read_fraction=fr, timeout=t)
            search(nodes, read_fraction=fr, timeout=t, optimize='network')
            search(nodes, read_fraction=fr, timeout=t, optimize='latency')
            search(nodes, read_fraction=fr, timeout=t, resilience=1)
            search(nodes, read_fraction=fr, timeout=t, f=1)
