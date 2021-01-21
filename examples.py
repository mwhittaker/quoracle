from quorums.quorums import *

a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')
g = Node('g')
h = Node('h')
i = Node('i')


def main():
    # TODO(mwhittaker): Add more quorums and tidy up.
    quorum_systems = {
        'majority': QuorumSystem(reads=majority([a, b, c])),
        'read one, write all': QuorumSystem(reads=choose(1, [a, b, c])),
        'write one, read all': QuorumSystem(writes=choose(1, [a, b, c])),
        '3 by 3 grid': QuorumSystem(reads=a*b*c + d*e*f + g*h*i),
    }

    for name, qs in quorum_systems.items():
        sigma = qs.strategy(read_fraction=0.5)
        print(name)
        print(qs)
        print(sigma)
        print(sigma.load(read_fraction=0.5))

if __name__ == '__main__':
    main()
