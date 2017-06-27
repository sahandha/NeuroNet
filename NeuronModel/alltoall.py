from mpi4py import MPI
comm = MPI.COMM_WORLD

cs = comm.size
r = comm.rank

# data in all processes
data = []


data += [r*cs + i for i in range(cs)]



# print input data
msg = "[%d] input:  %s" % (r, data)
print(msg)

comm.Barrier()

# alltoall
data = comm.alltoall(data)


# print result data
msg = "[%d] result: %s" % (r, data)
print(msg)
