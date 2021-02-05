from quoracle import *
from quoracle.quorum_system import *
import unittest


class TestStrategy(unittest.TestCase):
    def test_get_quorum(self) -> None:
        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')
        e = Node('e')

        for reads in [a,
                      a + b,
                      a + b + c,
                      choose(2, [a, b, c]),
                      choose(2, [a, b, c, d, e]),
                      (a + b) * (c + d),
                      (a * b) + (c * d)]:
            qs = QuorumSystem(reads=reads)
            sigma = qs.uniform_strategy()
            for _ in range(10):
                self.assertTrue(qs.is_read_quorum(sigma.get_read_quorum()))
                self.assertTrue(qs.is_write_quorum(sigma.get_write_quorum()))

    def test_load_cap_util(self) -> None:
        a = Node('a', write_capacity=10, read_capacity=50)
        b = Node('b', write_capacity=20, read_capacity=60)
        c = Node('c', write_capacity=30, read_capacity=70)
        d = Node('d', write_capacity=40, read_capacity=80)

        qs = QuorumSystem(reads=a*b + c*d)
        sigma = qs.make_strategy(
            sigma_r = {
                frozenset({'a', 'b'}): 0.75,
                frozenset({'c', 'd'}): 0.25,
            },
            sigma_w = {
                frozenset({'a', 'c'}): 0.1,
                frozenset({'a', 'd'}): 0.2,
                frozenset({'b', 'c'}): 0.3,
                frozenset({'b', 'd'}): 0.4,
            },
        )

        node_loads_08 = {
            a: 0.8 / 50 * 0.75 + 0.2 / 10 * (0.1 + 0.2),
            b: 0.8 / 60 * 0.75 + 0.2 / 20 * (0.3 + 0.4),
            c: 0.8 / 70 * 0.25 + 0.2 / 30 * (0.1 + 0.3),
            d: 0.8 / 80 * 0.25 + 0.2 / 40 * (0.2 + 0.4),
        }
        load_08 = max(node_loads_08.values())
        cap_08 = 1 / load_08
        node_throughputs_08 = {
            a: cap_08 * (0.8 * 0.75 + 0.2 * (0.1 + 0.2)),
            b: cap_08 * (0.8 * 0.75 + 0.2 * (0.3 + 0.4)),
            c: cap_08 * (0.8 * 0.25 + 0.2 * (0.1 + 0.3)),
            d: cap_08 * (0.8 * 0.25 + 0.2 * (0.2 + 0.4)),
        }
        self.assertAlmostEqual(sigma.load(read_fraction=0.8), load_08)
        self.assertAlmostEqual(sigma.capacity(read_fraction=0.8), cap_08)
        for node, l in node_loads_08.items():
            self.assertAlmostEqual(sigma.node_load(node, read_fraction=0.8), l)
            self.assertAlmostEqual(
                sigma.node_utilization(node, read_fraction=0.8),
                l * cap_08)
        for node, t in node_throughputs_08.items():
            self.assertAlmostEqual(
                sigma.node_throughput(node, read_fraction=0.8),
                t)

        node_loads_05 = {
            a: 0.5 / 50 * 0.75 + 0.5 / 10 * (0.1 + 0.2),
            b: 0.5 / 60 * 0.75 + 0.5 / 20 * (0.3 + 0.4),
            c: 0.5 / 70 * 0.25 + 0.5 / 30 * (0.1 + 0.3),
            d: 0.5 / 80 * 0.25 + 0.5 / 40 * (0.2 + 0.4),
        }
        load_05 = max(node_loads_05.values())
        cap_05 = 1 / load_05
        node_throughputs_05 = {
            a: cap_05 * (0.5 * 0.75 + 0.5 * (0.1 + 0.2)),
            b: cap_05 * (0.5 * 0.75 + 0.5 * (0.3 + 0.4)),
            c: cap_05 * (0.5 * 0.25 + 0.5 * (0.1 + 0.3)),
            d: cap_05 * (0.5 * 0.25 + 0.5 * (0.2 + 0.4)),
        }
        self.assertAlmostEqual(sigma.load(read_fraction=0.5), load_05)
        self.assertAlmostEqual(sigma.capacity(read_fraction=0.5), cap_05)
        for node, l in node_loads_05.items():
            self.assertAlmostEqual(sigma.node_load(node, read_fraction=0.5), l)
            self.assertAlmostEqual(
                sigma.node_utilization(node, read_fraction=0.5),
                l * cap_05)
        for node, t in node_throughputs_05.items():
            self.assertAlmostEqual(
                sigma.node_throughput(node, read_fraction=0.5),
                t)

        fr = {0.8: 0.7, 0.5: 0.3}
        node_loads = {
            a: 0.7 * (0.8 / 50 * 0.75 + 0.2 / 10 * (0.1 + 0.2)) +
               0.3 * (0.5 / 50 * 0.75 + 0.5 / 10 * (0.1 + 0.2)),
            b: 0.7 * (0.8 / 60 * 0.75 + 0.2 / 20 * (0.3 + 0.4)) +
               0.3 * (0.5 / 60 * 0.75 + 0.5 / 20 * (0.3 + 0.4)),
            c: 0.7 * (0.8 / 70 * 0.25 + 0.2 / 30 * (0.1 + 0.3)) +
               0.3 * (0.5 / 70 * 0.25 + 0.5 / 30 * (0.1 + 0.3)),
            d: 0.7 * (0.8 / 80 * 0.25 + 0.2 / 40 * (0.2 + 0.4)) +
               0.3 * (0.5 / 80 * 0.25 + 0.5 / 40 * (0.2 + 0.4)),
        }
        load = (0.7 * max(node_loads_08.values()) +
                0.3 * max(node_loads_05.values()))
        cap = (0.7 * 1 / max(node_loads_08.values()) +
                0.3 * 1/ max(node_loads_05.values()))
        self.assertAlmostEqual(sigma.load(read_fraction=fr), load)
        self.assertAlmostEqual(sigma.capacity(read_fraction=fr), cap)
        node_throughputs = {
            a: cap_08 * 0.7 * (0.8 * 0.75 + 0.2 * (0.1 + 0.2)) +
               cap_05 * 0.3 * (0.5 * 0.75 + 0.5 * (0.1 + 0.2)),
            b: cap_08 * 0.7 * (0.8 * 0.75 + 0.2 * (0.3 + 0.4)) +
               cap_05 * 0.3 * (0.5 * 0.75 + 0.5 * (0.3 + 0.4)),
            c: cap_08 * 0.7 * (0.8 * 0.25 + 0.2 * (0.1 + 0.3)) +
               cap_05 * 0.3 * (0.5 * 0.25 + 0.5 * (0.1 + 0.3)),
            d: cap_08 * 0.7 * (0.8 * 0.25 + 0.2 * (0.2 + 0.4)) +
               cap_05 * 0.3 * (0.5 * 0.25 + 0.5 * (0.2 + 0.4)),
        }
        for node, l in node_loads.items():
            self.assertAlmostEqual(sigma.node_load(node, read_fraction=fr), l)
            self.assertAlmostEqual(
                sigma.node_utilization(node, read_fraction=fr),
                0.7 * cap_08 * node_loads_08[node] +
                0.3 * cap_05 * node_loads_05[node])
        for node, t in node_throughputs.items():
            self.assertAlmostEqual(
                sigma.node_throughput(node, read_fraction=fr),
                t)

    def test_network_load(self) -> None:
        a = Node('a')
        b = Node('b')
        c = Node('c')
        d = Node('d')
        e = Node('e')

        qs = QuorumSystem(reads=a*b + c*d*e)
        sigma = qs.make_strategy(
            sigma_r = {
                frozenset({'a', 'b'}): 75,
                frozenset({'c', 'd', 'e'}): 25,
            },
            sigma_w = {
                frozenset({'a', 'c'}): 5,
                frozenset({'a', 'd'}): 10,
                frozenset({'a', 'e'}): 15,
                frozenset({'b', 'c'}): 20,
                frozenset({'b', 'd'}): 25,
                frozenset({'b', 'e'}): 25,
            },
        )

        self.assertEqual(sigma.network_load(read_fraction=0.8),
            0.8 * 0.75 * 2 +
            0.8 * 0.25 * 3 +
            0.2 * 2
        )

    def test_latency(self) -> None:
        a = Node('a', latency=datetime.timedelta(seconds=1))
        b = Node('b', latency=datetime.timedelta(seconds=2))
        c = Node('c', latency=datetime.timedelta(seconds=3))
        d = Node('d', latency=datetime.timedelta(seconds=4))
        e = Node('e', latency=datetime.timedelta(seconds=5))

        qs = QuorumSystem(reads=a*b + c*d*e)
        sigma = qs.make_strategy(
            sigma_r = {
                frozenset({'a', 'b'}): 10,
                frozenset({'a', 'b', 'c'}): 20,
                frozenset({'c', 'd', 'e'}): 30,
                frozenset({'c', 'd', 'e', 'a'}): 40,
            },
            sigma_w = {
                frozenset({'a', 'c'}): 5,
                frozenset({'a', 'd'}): 10,
                frozenset({'a', 'e'}): 15,
                frozenset({'b', 'c'}): 20,
                frozenset({'b', 'd'}): 25,
                frozenset({'b', 'e'}): 25,
            },
        )

        self.assertEqual(sigma.latency(read_fraction=0.8),
            0.8 * 0.10 * datetime.timedelta(seconds=2) +
            0.8 * 0.20 * datetime.timedelta(seconds=2) +
            0.8 * 0.30 * datetime.timedelta(seconds=5) +
            0.8 * 0.40 * datetime.timedelta(seconds=5) +
            0.2 * 0.05 * datetime.timedelta(seconds=3) +
            0.2 * 0.10 * datetime.timedelta(seconds=4) +
            0.2 * 0.15 * datetime.timedelta(seconds=5) +
            0.2 * 0.20 * datetime.timedelta(seconds=3) +
            0.2 * 0.25 * datetime.timedelta(seconds=4) +
            0.2 * 0.25 * datetime.timedelta(seconds=5)
        )
