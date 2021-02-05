from quoracle import *
from quoracle.geometry import *
from typing import Any, Callable, List, NamedTuple, Optional, Tuple
import unittest


class TestGeometry(unittest.TestCase):
    def test_eq(self):
        l = Point(0, 1)
        r = Point(1, 1)
        m = Point(0.5, 0.5)
        self.assertEqual(Segment(l, r), Segment(l, r))
        self.assertNotEqual(Segment(l, r), Segment(l, m))

    def test_compatible(self):
        s1 = Segment(Point(0, 1), Point(1, 2))
        s2 = Segment(Point(0, 2), Point(1, 1))
        s3 = Segment(Point(0.5, 2), Point(1, 1))
        self.assertTrue(s1.compatible(s2))
        self.assertTrue(s2.compatible(s1))
        self.assertFalse(s1.compatible(s3))
        self.assertFalse(s3.compatible(s1))
        self.assertFalse(s2.compatible(s3))
        self.assertFalse(s3.compatible(s2))

    def test_call(self):
        segment = Segment(Point(0, 0), Point(1, 1))
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            self.assertEqual(segment(x), x)

        segment = Segment(Point(0, 0), Point(1, 2))
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            self.assertEqual(segment(x), 2*x)

        segment = Segment(Point(1, 2), Point(3, 6))
        for x in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
            self.assertEqual(segment(x), 2*x)

        segment = Segment(Point(0, 1), Point(1, 0))
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            self.assertEqual(segment(x), 1 - x)

    def test_slope(self):
        self.assertEqual(Segment(Point(0, 0), Point(1, 1)).slope(), 1.0)
        self.assertEqual(Segment(Point(0, 1), Point(1, 2)).slope(), 1.0)
        self.assertEqual(Segment(Point(1, 1), Point(2, 2)).slope(), 1.0)
        self.assertEqual(Segment(Point(1, 1), Point(2, 3)).slope(), 2.0)
        self.assertEqual(Segment(Point(1, 1), Point(2, 0)).slope(), -1.0)

    def test_above(self):
        s1 = Segment(Point(0, 0), Point(1, 0.5))
        s2 = Segment(Point(0, 0.5), Point(1, 2))
        s3 = Segment(Point(0, 1.5), Point(1, 0.5))

        self.assertFalse(s1.above(s1))
        self.assertFalse(s1.above(s2))
        self.assertFalse(s1.above(s3))

        self.assertTrue(s2.above(s1))
        self.assertFalse(s2.above(s2))
        self.assertFalse(s2.above(s3))

        self.assertTrue(s3.above(s1))
        self.assertFalse(s3.above(s2))
        self.assertFalse(s3.above(s3))

    def test_above_eq(self):
        s1 = Segment(Point(0, 0), Point(1, 0.5))
        s2 = Segment(Point(0, 0.5), Point(1, 2))
        s3 = Segment(Point(0, 1.5), Point(1, 0.5))

        self.assertTrue(s1.above_eq(s1))
        self.assertFalse(s1.above_eq(s2))
        self.assertFalse(s1.above_eq(s3))

        self.assertTrue(s2.above_eq(s1))
        self.assertTrue(s2.above_eq(s2))
        self.assertFalse(s2.above_eq(s3))

        self.assertTrue(s3.above_eq(s1))
        self.assertFalse(s3.above_eq(s2))
        self.assertTrue(s3.above_eq(s3))

    def test_intersects(self):
        s1 = Segment(Point(0, 0), Point(1, 0.5))
        s2 = Segment(Point(0, 0.5), Point(1, 2))
        s3 = Segment(Point(0, 1.5), Point(1, 0.5))

        self.assertTrue(s1.intersects(s1))
        self.assertFalse(s1.intersects(s2))
        self.assertTrue(s1.intersects(s3))

        self.assertFalse(s2.intersects(s1))
        self.assertTrue(s2.intersects(s2))
        self.assertTrue(s2.intersects(s3))

        self.assertTrue(s3.intersects(s1))
        self.assertTrue(s3.intersects(s2))
        self.assertTrue(s3.intersects(s3))

    def test_intersection(self):
        s1 = Segment(Point(0, 0), Point(1, 1))
        s2 = Segment(Point(0, 1), Point(1, 0))
        s3 = Segment(Point(0, 1), Point(1, 1))
        s4 = Segment(Point(0, 0.25), Point(1, 0.25))

        self.assertEqual(s1.intersection(s1), None)
        self.assertEqual(s1.intersection(s2), Point(0.5, 0.5))
        self.assertEqual(s1.intersection(s3), Point(1, 1))
        self.assertEqual(s1.intersection(s4), Point(0.25, 0.25))

        self.assertEqual(s2.intersection(s1), Point(0.5, 0.5))
        self.assertEqual(s2.intersection(s2), None)
        self.assertEqual(s2.intersection(s3), Point(0, 1))
        self.assertEqual(s2.intersection(s4), Point(0.75, 0.25))

        self.assertEqual(s3.intersection(s1), Point(1, 1))
        self.assertEqual(s3.intersection(s2), Point(0, 1))
        self.assertEqual(s3.intersection(s3), None)
        self.assertEqual(s3.intersection(s4), None)

        self.assertEqual(s4.intersection(s1), Point(0.25, 0.25))
        self.assertEqual(s4.intersection(s2), Point(0.75, 0.25))
        self.assertEqual(s4.intersection(s3), None)
        self.assertEqual(s4.intersection(s4), None)

    def test_max_one_segment(self):
        s1 = Segment(Point(0, 0), Point(1, 1))
        s2 = Segment(Point(0, 1), Point(1, 0))
        s3 = Segment(Point(0, 1), Point(1, 1))
        s4 = Segment(Point(0, 0.25), Point(1, 0.25))
        s5 = Segment(Point(0, 0.75), Point(1, 0.75))

        def is_subset(xs: List[Any], ys: List[Any]) -> bool:
            return all(x in ys for x in xs)

        for s in [s1, s2, s3, s4, s5]:
            self.assertEqual(max_of_segments([s]), [s.l, s.r])

        expected = [
            ([s1, s1], [(0, 0), (1, 1)]),
            ([s1, s2], [(0, 1), (0.5, 0.5), (1, 1)]),
            ([s1, s3], [(0, 1), (1, 1)]),
            ([s1, s4], [(0, 0.25), (0.25, 0.25), (1, 1)]),
            ([s1, s5], [(0, 0.75), (0.75, 0.75), (1, 1)]),
            ([s2, s2], [(0, 1), (1, 0)]),
            ([s2, s3], [(0, 1), (1, 1)]),
            ([s2, s4], [(0, 1), (0.75, 0.25), (1, 0.25)]),
            ([s2, s5], [(0, 1), (0.25, 0.75), (1, 0.75)]),
            ([s3, s3], [(0, 1), (1, 1)]),
            ([s3, s4], [(0, 1), (1, 1)]),
            ([s3, s5], [(0, 1), (1, 1)]),
            ([s4, s4], [(0, 0.25), (1, 0.25)]),
            ([s4, s5], [(0, 0.75), (1, 0.75)]),
            ([s5, s5], [(0, 0.75), (1, 0.75)]),

            ([s1, s2, s4], [(0, 1), (0.5, 0.5), (1, 1)]),
            ([s1, s2, s5], [(0, 1), (0.25, 0.75), (0.75, 0.75), (1, 1)]),
        ]
        for segments, path in expected:
            self.assertTrue(is_subset(path, max_of_segments(segments)))
            self.assertTrue(is_subset(path, max_of_segments(segments[::-1])))
