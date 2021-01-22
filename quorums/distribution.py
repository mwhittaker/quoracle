from typing import Dict, Optional, Union

Fraction = float
Weight = float
Probability = float
Distribution = Union[
    # For example, 1 means 100% reads.
    int,
    # For example, 0.25 means 25% reads.
    float,
    # For example, {0.25: 1, 0.8: 2} means 25% reads one third of the time and
    # 80% reads two thirds of the time.
    Dict[Fraction, Weight],
]

def canonicalize(d: Distribution) -> Dict[Fraction, Probability]:
    if isinstance(d, int):
        if d < 0 or d > 1:
            raise ValueError('distribution must be in the range [0, 1]')
        return {float(d): 1.}
    elif isinstance(d, float):
        if d < 0 or d > 1:
            raise ValueError('distribution must be in the range [0, 1]')
        return {d: 1.}
    elif isinstance(d, dict):
        if len(d) == 0:
            raise ValueError('distribution cannot empty')

        if any(weight < 0 for weight in d.values()):
            raise ValueError('distribution cannot have negative weights')

        total_weight = sum(d.values())
        if total_weight == 0:
            raise ValueError('distribution cannot have zero weight')

        return {float(f): weight / total_weight
                for (f, weight) in d.items()
                if weight > 0}
    else:
        raise ValueError('distribution must be an int, a float, a Dict[float, '
                         'float] or a List[Tuple[float, float]]')


def canonicalize_rw(read_fraction: Optional[Distribution],
                    write_fraction: Optional[Distribution]) \
                    -> Dict[Fraction, Probability]:
    if read_fraction is None and write_fraction is None:
        raise ValueError('Either read_fraction or write_fraction must be given')
    elif read_fraction is not None and write_fraction is not None:
        raise ValueError('Only one of read_fraction or write_fraction can be '
                         'given')
    elif read_fraction is not None:
        return canonicalize(read_fraction)
    else:
        assert write_fraction is not None
        return {1 - f: weight
                for (f, weight) in canonicalize(write_fraction).items()}
