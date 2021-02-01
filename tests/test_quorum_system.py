from quorums import *
from quorums.quorum_system import *
import unittest

class TestExpr(unittest.TestCase):
    def test_init(self):
        def quorums(e: Expr['str']) -> FrozenSet[FrozenSet[str]]:
            return frozenset(frozenset(q) for q in e.quorums())

        def assert_equal(x: Expr['str'], y: Expr['str']) -> None:
            self.assertEqual(quorums(x), quorums(y))

        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')

        # Specify reads.
        qs = QuorumSystem(reads = a + b)
        assert_equal(qs.reads, a + b)
        assert_equal(qs.writes, a * b)

        # Specify writes.
        qs = QuorumSystem(writes = a + b)
        assert_equal(qs.reads, a * b)
        assert_equal(qs.writes, a + b)

        # Specify neither.
        with self.assertRaises(ValueError):
            QuorumSystem()

        # Specify both overlapping.
        qs = QuorumSystem(reads=a+b, writes=a*b*c)
        assert_equal(qs.reads, a+b)
        assert_equal(qs.writes, a*b*c)

        # Specify both not overlapping.
        with self.assertRaises(ValueError):
            QuorumSystem(reads=a+b, writes=a)

    def test_uniform_strategy(self):
        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')

        sigma = QuorumSystem(reads=a).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a'}): 1.0,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a'}): 1.0,
        })

        sigma = QuorumSystem(reads=a+a).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a'}): 1.0,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a'}): 1.0,
        })

        sigma = QuorumSystem(reads=a*a).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a'}): 1.0,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a'}): 1.0,
        })

        sigma = QuorumSystem(reads=a + a*b).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a'}): 1.0,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a'}): 1.0,
        })

        sigma = QuorumSystem(reads=a + a*b + a*c).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a'}): 1.0,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a'}): 1.0,
        })

        sigma = QuorumSystem(reads=a + b).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a'}): 1 / 2,
            frozenset({'b'}): 1 / 2,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a', 'b'}): 1.0,
        })

        sigma = QuorumSystem(reads=a + b + c).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a'}): 1 / 3,
            frozenset({'b'}): 1 / 3,
            frozenset({'c'}): 1 / 3,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a', 'b', 'c'}): 1.0,
        })

        sigma = QuorumSystem(reads=(a*b)+(c*d)).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a', 'b'}): 1 / 2,
            frozenset({'c', 'd'}): 1 / 2,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a', 'c'}): 1 / 4,
            frozenset({'a', 'd'}): 1 / 4,
            frozenset({'b', 'c'}): 1 / 4,
            frozenset({'b', 'd'}): 1 / 4,
        })

        sigma = QuorumSystem(reads=(a*b)+(c*d)+(a*b)+(a*b*c)).uniform_strategy()
        self.assertEqual(sigma.sigma_r, {
            frozenset({'a', 'b'}): 1 / 2,
            frozenset({'c', 'd'}): 1 / 2,
        })
        self.assertEqual(sigma.sigma_w, {
            frozenset({'a', 'c'}): 1 / 4,
            frozenset({'a', 'd'}): 1 / 4,
            frozenset({'b', 'c'}): 1 / 4,
            frozenset({'b', 'd'}): 1 / 4,
        })

    def test_optimal_strategy(self):
        # TODO(mwhittaker): Implement.
        pass
