from quoracle import *
from quoracle.quorum_system import *
import unittest


class TestQuorumSystem(unittest.TestCase):
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

    def test_make_strategy(self):
        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')

        qs = QuorumSystem(reads = a*b + c*d)
        sigma = qs.make_strategy(
            sigma_r = {
                frozenset({'a', 'b'}): 25,
                frozenset({'c', 'd'}): 75,
            },
            sigma_w = {
                frozenset({'a', 'c'}): 1,
                frozenset({'a', 'd'}): 1,
                frozenset({'b', 'c'}): 1,
                frozenset({'b', 'd'}): 1,
            },
        )
        self.assertEqual(sigma.sigma_r,
            {
                frozenset({'a', 'b'}): 0.25,
                frozenset({'c', 'd'}): 0.75,
            },
        )
        self.assertEqual(sigma.sigma_w,
            {
                frozenset({'a', 'c'}): 0.25,
                frozenset({'a', 'd'}): 0.25,
                frozenset({'b', 'c'}): 0.25,
                frozenset({'b', 'd'}): 0.25,
            },
        )

        with self.assertRaises(ValueError):
            sigma = qs.make_strategy(
                sigma_r = {
                    frozenset({'a', 'b'}): -1,
                    frozenset({'c', 'd'}): 1,
                },
                sigma_w = {
                    frozenset({'a', 'c'}): 1,
                    frozenset({'a', 'd'}): 1,
                    frozenset({'b', 'c'}): 1,
                    frozenset({'b', 'd'}): 1,
                },
            )

        with self.assertRaises(ValueError):
            sigma = qs.make_strategy(
                sigma_r = {
                    frozenset({'a'}): 1,
                    frozenset({'c', 'd'}): 1,
                },
                sigma_w = {
                    frozenset({'a', 'c'}): 1,
                    frozenset({'a', 'd'}): 1,
                    frozenset({'b', 'c'}): 1,
                    frozenset({'b', 'd'}): 1,
                },
            )

    def test_optimal_strategy(self):
        def s(n: int) -> datetime.timedelta:
            return datetime.timedelta(seconds=n)

        a = Node('a', write_capacity=1, read_capacity=2, latency=s(1))
        b = Node('b', write_capacity=1, read_capacity=2, latency=s(2))
        c = Node('c', write_capacity=1, read_capacity=2, latency=s(3))
        d = Node('d', write_capacity=1, read_capacity=2, latency=s(4))
        qs = QuorumSystem(reads=a*b + c*d)

        # Load Optimized.
        self.assertEqual(qs.load(read_fraction=1), 0.25)
        self.assertEqual(qs.capacity(read_fraction=1), 4)
        self.assertEqual(qs.load(read_fraction=0), 0.5)
        self.assertEqual(qs.capacity(read_fraction=0), 2)

        self.assertEqual(qs.load(read_fraction=1, network_limit=2), 0.25)
        self.assertEqual(qs.capacity(read_fraction=1, network_limit=2), 4)
        self.assertEqual(qs.load(read_fraction=0, network_limit=2), 0.5)
        self.assertEqual(qs.capacity(read_fraction=0, network_limit=2), 2)

        self.assertEqual(qs.load(read_fraction=1, latency_limit=s(4)), 0.25)
        self.assertEqual(qs.capacity(read_fraction=1, latency_limit=s(4)), 4)
        self.assertEqual(qs.load(read_fraction=0, latency_limit=s(4)), 0.5)
        self.assertEqual(qs.capacity(read_fraction=0, latency_limit=s(4)), 2)

        # Network Optimized.
        self.assertEqual(qs.network_load(
            read_fraction=1,
            optimize='network',
        ), 2)
        self.assertEqual(qs.network_load(
            read_fraction=0,
            optimize='network',
        ), 2)
        self.assertEqual(qs.network_load(
            read_fraction=1,
            optimize='network',
            load_limit = 0.25,
        ), 2)
        self.assertEqual(qs.network_load(
            read_fraction=0,
            optimize='network',
            load_limit = 0.5,
        ), 2)
        self.assertEqual(qs.network_load(
            read_fraction=1,
            optimize='network',
            latency_limit = s(2),
        ), 2)
        self.assertEqual(qs.network_load(
            read_fraction=0,
            optimize='network',
            latency_limit = s(3),
        ), 2)

        # Latency Optimized.
        self.assertEqual(qs.latency(read_fraction=1, optimize='latency'), s(2))
        self.assertEqual(qs.latency(read_fraction=0, optimize='latency'), s(3))
        self.assertEqual(qs.latency(
            read_fraction=1,
            optimize='latency',
            load_limit = 1.0,
        ), s(2))
        self.assertEqual(qs.latency(
            read_fraction=0,
            optimize='latency',
            load_limit = 1.0,
        ), s(3))
        self.assertEqual(qs.latency(
            read_fraction=1,
            optimize='latency',
            network_limit = 2,
        ), s(2))
        self.assertEqual(qs.latency(
            read_fraction=0,
            optimize='latency',
            network_limit = 2,
        ), s(3))

        # 1-Resilient Load Optimized.
        self.assertEqual(qs.load(read_fraction=1, f=1), 0.5)
        self.assertEqual(qs.capacity(read_fraction=1, f=1), 2)
        self.assertEqual(qs.load(read_fraction=0, f=1), 1)
        self.assertEqual(qs.capacity(read_fraction=0, f=1), 1)

        # 1-Resilient Network Optimized.
        self.assertEqual(
            qs.network_load(read_fraction=1, optimize='network', f=1), 4)
        self.assertEqual(
            qs.network_load(read_fraction=0, optimize='network', f=1), 4)

        # 1-Resilient Latency Optimized.
        self.assertEqual(
            qs.latency(read_fraction=1, optimize='latency', f=1), s(2))
        self.assertEqual(
            qs.latency(read_fraction=0, optimize='latency', f=1), s(3))

        # Illegal Specification
        with self.assertRaises(ValueError):
            qs.strategy(read_fraction=0.1, optimize='load', load_limit=1)

        with self.assertRaises(ValueError):
            qs.strategy(read_fraction=0.1, optimize='network', network_limit=2)

        with self.assertRaises(ValueError):
            qs.strategy(read_fraction=0.1, optimize='latency',
                        latency_limit=s(5))

        # Unsatisfiable Constraints
        with self.assertRaises(NoStrategyFoundError):
            qs.strategy(read_fraction=0,
                        optimize='load',
                        network_limit=1.5)

        with self.assertRaises(NoStrategyFoundError):
            qs.strategy(read_fraction=0,
                        optimize='load',
                        latency_limit=s(1))

        with self.assertRaises(NoStrategyFoundError):
            qs.strategy(read_fraction=1,
                        optimize='network',
                        load_limit=0.25,
                        latency_limit=s(2))
