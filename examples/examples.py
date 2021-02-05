from quoracle import *


def load(qs: QuorumSystem, fr: float, f: int) -> float:
    try:
        return qs.load(read_fraction=fr, f=f)
    except ValueError:
        return float('inf')


def main():
    a = Node('a')
    b = Node('b')
    c = Node('c')
    d = Node('d')
    e = Node('e')

    reads_examples = [
        # 1 node.
        a,

        # 2 nodes.
        choose(1, [a, b]),
        choose(2, [a, b]),

        # 3 nodes.
        choose(1, [a, b, c]),
        choose(2, [a, b, c]),
        choose(3, [a, b, c]),

        # 4 nodes.
        a*b + c*d,
        (a+b)*(c+d),
        choose(1, [a, b, c, d]),
        choose(2, [a, b, c, d]),
        choose(3, [a, b, c, d]),
        choose(4, [a, b, c, d]),

        # 5 nodes.
        a*b + a*c*e + d*e + d*c*b,
        a*b + c*d*e,
        (a+b) * (c+d+e),
        (a+b) * (c+d+e),
        a + b*c + d*e,
        a * (b+c) * (d+e),
        choose(1, [a, b, c, d, e]),
        choose(2, [a, b, c, d, e]),
        choose(3, [a, b, c, d, e]),
        choose(4, [a, b, c, d, e]),
        choose(5, [a, b, c, d, e]),
    ]

    fs = [0, 1, 2]
    frs = [0, 0.25, 0.5, 0.75, 1]
    header = (['Quorum System', 'n', 'Dup Free?', 'Read Resilience',
               'Write Resilience', 'Resilience'] +
              [f'f={f},fr={fr}' for f in fs for fr in frs])
    print(';'.join(header))

    for reads in reads_examples:
        qs = QuorumSystem(reads=reads)
        data = ([reads, len(qs.nodes()), qs.dup_free(), qs.read_resilience(),
                 qs.write_resilience(), qs.resilience()]+
                ['{:.4f}'.format(load(qs, fr, f=f))
                 for f in [0, 1, 2]
                 for fr in [0, 0.25, 0.5, 0.75, 1]])
        print(';'.join(str(x) for x in data))


if __name__ == '__main__':
    main()
