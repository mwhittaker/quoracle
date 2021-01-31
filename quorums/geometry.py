from typing import Any, Callable, List, NamedTuple, Optional, Tuple
import math


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

    def __hash__(self) -> int:
        return hash((self.l, self.r))

    def __call__(self, x: float) -> float:
        assert self.l.x <= x <= self.r.x
        return self.slope() * (x - self.l.x) + self.l.y

    def approximately_equal(self, other: 'Segment') -> float:
        return (math.isclose(self.l.y, other.l.y, rel_tol=1e-5) and
                math.isclose(self.r.y, other.r.y, rel_tol=1e-5))

    def compatible(self, other: 'Segment') -> float:
        return self.l.x == other.l.x and self.r.x == other.r.x

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

    # We compute the x-coordinate of every intersection point. We sort the
    # x-coordinates and for every x, we compute the highest line at that point.
    xs: List[float] = [0.0, 1.0]
    for (i, s1) in enumerate(segments):
        for (j, s2) in enumerate(segments[i + 1:], i + 1):
            p = s1.intersection(s2)
            if p is not None:
                xs.append(p.x)
    xs.sort()
    return [(x, max(segments, key=lambda s: s(x))(x)) for x in xs]
