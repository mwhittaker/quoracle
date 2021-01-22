# TODO(mwhittaker): We can define a set of read quorums that are not minimal.
# Does this mess things up?



# a = Node('a')
# b = Node('b')
# c = Node('c')
# d = Node('d')
# e = Node('e')
# f = Node('f')
# g = Node('g')
# h = Node('h')
# i = Node('i')
#
# walls = QuorumSystem(reads=a*b + c*d*e)
# paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)
# maj = QuorumSystem(reads=majority([a, b, c, d, e]))
#
# for qs in [walls, paths, maj]:
#     print(qs.dup_free())
#     print(qs.resilience())

    # sigma_0 = qs.strategy(read_fraction=0.5)
    # sigma_1 = qs.strategy(read_fraction=0.5, f=1)
    # print(sigma_0.load(read_fraction=0.5), sigma_1.load(read_fraction=0.5))
    # print(sigma_1)


#
# qs = QuorumSystem(reads = a*b + a*c)
# print(list(qs.read_quorums()))
# sigma = qs.strategy(read_fraction=0.5)
# print(list(qs.write_quorums()))
# print(sigma)
# print(1 / sigma.load(read_fraction=0.5))

# paths = QuorumSystem(reads=a*b + a*c*e + d*e + d*c*b)
# print(paths.resilience())
# sigma = paths.strategy(read_fraction=0.5)
# print(sigma.load(read_fraction=0.5))
#
# walls = QuorumSystem(reads=a*b + c*d*e)
# print(walls.resilience())
# sigma = walls.strategy(read_fraction=0.5)
# print(sigma.load(read_fraction=0.5))



# wpaxos = QuorumSystem(reads=majority([majority([a, b, c]),
#                                       majority([d, e, f]),
#                                       majority([g, h, i])]))
# sigma_1 = wpaxos.strategy(read_fraction=0.1)
# sigma_5 = wpaxos.strategy(read_fraction=0.5)
# sigma_9 = wpaxos.strategy(read_fraction=0.9)
# sigma_even = wpaxos.strategy(read_fraction={0.1: 2, 0.5: 2, 0.9: 1})
# for sigma in [sigma_1, sigma_5, sigma_9, sigma_even]:
#     frs = [0.1, 0.5, 0.9, {0.1: 2, 0.5: 2, 0.9: 1}]
#     print([sigma.load(fr) for fr in frs])

# - num_quorums
# - has dups?
# - optimal schedule
# - independent schedule
# - node read and write throughputs
