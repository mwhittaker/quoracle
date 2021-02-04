from quorums import *

a = Node('a')
b = Node('b')
c = Node('c')
d = Node('d')
e = Node('e')
f = Node('f')

grid = QuorumSystem(reads=a*b*c + d*e*f)

for r in grid.read_quorums():
    print(r)

for w in grid.write_quorums():
    print(w)

QuorumSystem(writes=(a + b + c) * (d + e + f))

QuorumSystem(reads=a*b*c + d*e*f, writes=(a + b + c) * (d + e + f))

print(grid.is_read_quorum({'a', 'b', 'c'}))       # True
print(grid.is_read_quorum({'a', 'b', 'c', 'd'}))  # True
print(grid.is_read_quorum({'a', 'b', 'd'}))       # False

print(grid.is_write_quorum({'a', 'd'}))      # True
print(grid.is_write_quorum({'a', 'd', 'd'})) # True
print(grid.is_write_quorum({'a', 'b'}))      # False

print(grid.read_resilience())  # 1
print(grid.write_resilience()) # 2
print(grid.resilience())       # 1

strategy = grid.strategy(read_fraction=0.75)

print(strategy.get_read_quorum())
print(strategy.get_read_quorum())
print(strategy.get_read_quorum())
print(strategy.get_write_quorum())
print(strategy.get_write_quorum())
print(strategy.get_write_quorum())

print(strategy.load(read_fraction=0.75)) # 0.458

print(strategy.load(read_fraction=0))   # 0.333
print(strategy.load(read_fraction=0.5)) # 0.416
print(strategy.load(read_fraction=1))   # 0.5

print(grid.load(read_fraction=0.25)) # 0.375

distribution = {0.1: 0.5, 0.75: 0.5}
strategy = grid.strategy(read_fraction=distribution)
print(strategy.load(read_fraction=distribution)) # 0.404

strategy = grid.strategy(write_fraction=0.75)
print(strategy.load(write_fraction=distribution)) # 0.429

a = Node('a', capacity=1000)
b = Node('b', capacity=500)
c = Node('c', capacity=1000)
d = Node('d', capacity=500)
e = Node('e', capacity=1000)
f = Node('f', capacity=500)

grid = QuorumSystem(reads=a*b*c + d*e*f)
strategy = grid.strategy(read_fraction=0.75)
print(strategy.load(read_fraction=0.75))     # 0.00075
print(strategy.capacity(read_fraction=0.75)) # 1333

a = Node('a', write_capacity=1000, read_capacity=10000)
b = Node('b', write_capacity=500, read_capacity=5000)
c = Node('c', write_capacity=1000, read_capacity=10000)
d = Node('d', write_capacity=500, read_capacity=5000)
e = Node('e', write_capacity=1000, read_capacity=10000)
f = Node('f', write_capacity=500, read_capacity=5000)

grid = QuorumSystem(reads=a*b*c + d*e*f)
print(grid.capacity(read_fraction=1))   # 10,000
print(grid.capacity(read_fraction=0.5)) # 3913
print(grid.capacity(read_fraction=0))   # 2000

strategy = grid.strategy(read_fraction=0.5, f=1)

print(strategy.get_read_quorum())
print(strategy.get_write_quorum())

simple_majority = QuorumSystem(reads=majority([a, b, c, d, e]))
crumbling_walls = QuorumSystem(reads=a*b + c*d*e)
paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)

assert(simple_majority.resilience() >= 1)
assert(crumbling_walls.resilience() >= 1)
assert(paths.resilience() >= 1)

distribution = {0.9: 0.9, 0.1: 0.1}
print(simple_majority.capacity(read_fraction=distribution)) # 5089
print(crumbling_walls.capacity(read_fraction=distribution)) # 5824
print(paths.capacity(read_fraction=distribution))           # 5725

print(simple_majority.capacity(read_fraction=distribution, f=1)) # 3816
print(crumbling_walls.capacity(read_fraction=distribution, f=1)) # 1908
print(paths.capacity(read_fraction=distribution, f=1))           # 1908
