from typing import Any, Callable, List, NamedTuple, Optional, Tuple
import unittest


class Point(NamedTuple):
    x: float
    y: float


class Segment:
    def __init__(self, l: Point, r: Point) -> None:
        assert l != r
        assert l.x < r.x
        self.l = l
        self.r = r

    def __str__(self) -> str:
        return f'{tuple(self.l)} -> {tuple(self.r)}'

    def __repr__(self) -> str:
        return f'Segment({self.l}, {self.r})'

    def __eq__(self, other) -> bool:
        if isinstance(other, Segment):
            return (self.l, self.r) == (other.l, other.r)
        else:
            return False

    def compatible(self, other: 'Segment') -> float:
        return self.l.x == other.l.x and self.r.x == other.r.x

    def __call__(self, x: float) -> float:
        assert self.l.x <= x <= self.r.x
        return self.slope() * (x - self.l.x) + self.l.y

    def slope(self) -> float:
        return (self.r.y - self.l.y) / (self.r.x - self.l.x)

    def above(self, other: 'Segment') -> bool:
        assert self.compatible(other)
        return self != other and self.l.y >= other.l.y and self.r.y >= other.r.y

    def above_eq(self, other: 'Segment') -> bool:
        assert self.compatible(other)
        return self == other or self.above(other)

    def intersects(self, other: 'Segment') -> bool:
        assert self.compatible(other)

        if self == other:
            return True
        elif self.l.y == other.l.y or self.r.y == other.r.y:
            return True
        elif self.above(other) or other.above(self):
            return False
        else:
            return True

    def intersection(self, other: 'Segment') -> Optional[Point]:
        assert self.compatible(other)

        if self == other or not self.intersects(other):
            return None

        x = ((other.l.y - self.l.y) /
             (self.r.y - other.r.y + other.l.y - self.l.y))
        return Point(x, self(x))


def max_of_segments(segments: List[Segment]) -> List[Tuple[float, float]]:
    assert len(segments) > 0
    assert len({segment.l.x for segment in segments}) == 1
    assert len({segment.r.x for segment in segments}) == 1

    # First, we remove any segments that are subsumed by other segments.
    non_dominated: List[Segment] = []
    for segment in segments:
        if any(other.above_eq(segment) for other in non_dominated):
            # If this segment is dominated by another, we exclude it.
            pass
        else:
            # Otherwise, we add this segment and remove any that it dominates.
            non_dominated = [other
                             for other in non_dominated
                             if not segment.above_eq(other)]
            non_dominated.append(segment)

    # Next, we start at the leftmost segment and continually jump over to the
    # segment with the first intersection.
    segment = max(non_dominated, key=lambda segment: segment.l.y)
    path: List[Point] = [segment.l]
    while True:
        intersections: List[Tuple[Point, Segment]] = []
        for other in non_dominated:
            p = segment.intersection(other)
            if p is not None and p.x > path[-1].x:
                intersections.append((p, other))

        if len(intersections) == 0:
            path.append(segment.r)
            return [(p.x, p.y) for p in path]

        intersection, segment = min(intersections, key=lambda t: t[0].x)
        path.append(intersection)


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
            self.assertEqual(max_of_segments(segments), path, segments)
            self.assertEqual(max_of_segments(segments[::-1]), path, segments)


if __name__ == '__main__':
    unittest.main()
