from quoracle import *
from quoracle.expr import *
from typing import Any, FrozenSet
import unittest

class TestExpr(unittest.TestCase):
    def test_quorums(self):
        def assert_equal(e: Expr[str], xs: List[Set[str]]) -> None:
            self.assertEqual(frozenset(frozenset(q) for q in e.quorums()),
                             frozenset(frozenset(x) for x in xs))
        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')
        e = Node('e')
        f = Node('f')

        assert_equal(a+b+c, [{'a'}, {'b'}, {'c'}])
        assert_equal(a*b*c, [{'a', 'b', 'c'}])
        assert_equal(a + b*c, [{'a'}, {'b', 'c'}])
        assert_equal(a*a*a, [{'a'}])
        assert_equal(a+a+a, [{'a'}])
        assert_equal(a*(a+b), [{'a'}, {'a', 'b'}])
        assert_equal(choose(1, [a, b, c]), [{'a'}, {'b'}, {'c'}])
        assert_equal(choose(2, [a, b, c]), [{'a', 'b'}, {'a', 'c'}, {'b', 'c'}])
        assert_equal(choose(3, [a, b, c]), [{'a', 'b', 'c'}])
        assert_equal((a+b) * (c+d), [
            {'a', 'c'}, {'a', 'd'}, {'b', 'c'}, {'b', 'd'}
        ])
        assert_equal((a+b) * (a+c), [
            {'a'}, {'a', 'c'}, {'a', 'b'}, {'b', 'c'}
        ])
        assert_equal(
            choose(2, [
                choose(2, [a, b, c]),
                choose(2, [d, e, f]),
                choose(2, [a, c, e]),
            ]), [
                {'a','b','d','e'}, {'a','b','d','f'}, {'a','b','e','f'},
                {'a','c','d','e'}, {'a','c','d','f'}, {'a','c','e','f'},
                {'b','c','d','e'}, {'b','c','d','f'}, {'b','c','e','f'},
                {'a','b','a','c'}, {'a','b','a','e'}, {'a','b','c','e'},
                {'a','c','a','c'}, {'a','c','a','e'}, {'a','c','c','e'},
                {'b','c','a','c'}, {'b','c','a','e'}, {'b','c','c','e'},
                {'d','e','a','c'}, {'d','e','a','e'}, {'d','e','c','e'},
                {'d','f','a','c'}, {'d','f','a','e'}, {'d','f','c','e'},
                {'e','f','a','c'}, {'e','f','a','e'}, {'e','f','c','e'},
        ])

    def test_is_quorum(self):
        def assert_quorum(e: Expr[str], q: Set[str]) -> None:
            self.assertTrue(e.is_quorum(q))

        def assert_not_quorum(e: Expr[str], q: Set[str]) -> None:
            self.assertFalse(e.is_quorum(q))

        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')
        e = Node('e')
        f = Node('f')

        expr = a + b + c
        assert_quorum(expr, {'a'})
        assert_quorum(expr, {'b'})
        assert_quorum(expr, {'c'})
        assert_quorum(expr, {'a', 'b'})
        assert_quorum(expr, {'a', 'c'})
        assert_quorum(expr, {'b', 'c'})
        assert_quorum(expr, {'a', 'b', 'c'})
        assert_quorum(expr, {'a', 'x'})
        assert_not_quorum(expr, set())
        assert_not_quorum(expr, {'x'})

        expr = a * b * c
        assert_quorum(expr, {'a', 'b', 'c'})
        assert_quorum(expr, {'a', 'b', 'c', 'x'})
        assert_not_quorum(expr, set())
        assert_not_quorum(expr, {'a'})
        assert_not_quorum(expr, {'b'})
        assert_not_quorum(expr, {'c'})
        assert_not_quorum(expr, {'a', 'b'})
        assert_not_quorum(expr, {'a', 'c'})
        assert_not_quorum(expr, {'b', 'c'})
        assert_not_quorum(expr, {'x'})
        assert_not_quorum(expr, {'a', 'x'})

        expr = choose(2, [a, b, c])
        assert_quorum(expr, {'a', 'b'})
        assert_quorum(expr, {'a', 'c'})
        assert_quorum(expr, {'b', 'c'})
        assert_quorum(expr, {'a', 'b', 'c'})
        assert_quorum(expr, {'a', 'b', 'c', 'x'})
        assert_not_quorum(expr, {'a'})
        assert_not_quorum(expr, {'b'})
        assert_not_quorum(expr, {'c'})
        assert_not_quorum(expr, {'x'})

        expr = (a+b) * (c+d)
        assert_quorum(expr, {'a', 'c'})
        assert_quorum(expr, {'a', 'c'})
        assert_quorum(expr, {'a', 'd'})
        assert_quorum(expr, {'b', 'd'})
        assert_quorum(expr, {'a', 'b', 'd'})
        assert_quorum(expr, {'b', 'c', 'd'})
        assert_quorum(expr, {'a', 'b', 'd'})
        assert_quorum(expr, {'a', 'c', 'd'})
        assert_quorum(expr, {'a', 'b', 'd'})
        assert_quorum(expr, {'b', 'c', 'd'})
        assert_quorum(expr, {'a', 'b', 'd'})
        assert_quorum(expr, {'b', 'c', 'd'})
        assert_quorum(expr, {'a', 'b', 'c', 'd'})
        assert_not_quorum(expr, {'a'})
        assert_not_quorum(expr, {'b'})
        assert_not_quorum(expr, {'c'})
        assert_not_quorum(expr, {'d'})
        assert_not_quorum(expr, {'a', 'b'})
        assert_not_quorum(expr, {'c', 'd'})
        assert_not_quorum(expr, {'a', 'b', 'x'})

    def test_resilience(self):
        def assert_resilience(e: Expr[str], n: int):
            self.assertEqual(e.resilience(), n)

        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')
        e = Node('e')
        f = Node('f')

        assert_resilience(a, 0)
        assert_resilience(a + b, 1)
        assert_resilience(a + b + c, 2)
        assert_resilience(a + b + c + d, 3)
        assert_resilience(a, 0)
        assert_resilience(a * b, 0)
        assert_resilience(a * b * c, 0)
        assert_resilience(a * b * c * d, 0)
        assert_resilience((a + b) * (c + d), 1)
        assert_resilience((a + b + c) * (d + e + f), 2)
        assert_resilience((a + b + c) * (a + e + f), 2)
        assert_resilience((a + a + c) * (d + e + f), 1)
        assert_resilience((a + a + a) * (d + e + f), 0)
        assert_resilience(a*b + b*c + a*d + a*d*e, 1)
        assert_resilience(choose(2, [a, b, c]), 1)
        assert_resilience(choose(2, [a, b, c, d, e]), 3)
        assert_resilience(choose(3, [a, b, c, d, e]), 2)
        assert_resilience(choose(4, [a, b, c, d, e]), 1)
        assert_resilience(choose(2, [a+b+c, d+e, f]), 2)
        assert_resilience(choose(2, [a*b, a*c, d]), 0)
        assert_resilience(choose(2, [a+b, a+c, a+d]), 2)

    def test_dual(self):
        def quorums(e: Expr['str']) -> FrozenSet[FrozenSet[str]]:
            return frozenset(frozenset(q) for q in e.quorums())

        def assert_equal(x: Expr['str'], y: Expr['str']) -> None:
            self.assertEqual(quorums(x), quorums(y))

        def assert_dual(x: Expr['str'], y: Expr['str']) -> None:
            assert_equal(x.dual(), y)

        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')
        e = Node('e')

        assert_dual(a, a)
        assert_dual(a + b, a * b)
        assert_dual(a + a, a * a)
        assert_dual((a + b) * (c + d), (a * b) + (c * d))
        assert_dual((a + b) * (a + d), (a * b) + (a * d))
        assert_dual((a + b) * (a + a), (a * b) + (a * a))
        assert_dual((a + a) * (a + a), (a * a) + (a * a))
        assert_dual((a + (a * b)) + ((c * d) + a),
                    (a * (a + b)) * ((c + d) * a))
        assert_dual(choose(2, [a, b, c]), choose(2, [a, b, c]))
        assert_dual(choose(2, [a+b, c+d, e]), choose(2, [a*b, c*d, e]))
        assert_dual(choose(3, [a, b, c, d, e]), choose(3, [a, b, c, d, e]))
        assert_dual(choose(2, [a, b, c, d, e]), choose(4, [a, b, c, d, e]))
        assert_dual(choose(4, [a, b, c, d, e]), choose(2, [a, b, c, d, e]))

    def test_dup_free(self):
        def assert_dup_free(e: Expr['str']) -> None:
            self.assertTrue(e.dup_free())

        def assert_not_dup_free(e: Expr['str']) -> None:
            self.assertFalse(e.dup_free())

        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')
        e = Node('e')
        f = Node('f')

        assert_dup_free(a)
        assert_dup_free(a + b)
        assert_dup_free(a * b)
        assert_dup_free(a * b + c)
        assert_dup_free(choose(2, [a, b, c]))
        assert_dup_free(choose(2, [a*b, c, d+e+f]))
        assert_dup_free(choose(3, [a, b, c, d, e]))
        assert_dup_free((a + b) * (c + (d * e)))

        assert_not_dup_free(a + a)
        assert_not_dup_free(a * a)
        assert_not_dup_free(a * (b + a))
        assert_not_dup_free(choose(2, [a, b, a]))
        assert_not_dup_free(choose(3, [a, b, c, d, a]))
        assert_not_dup_free((a + b) * (c + (d * a)))
