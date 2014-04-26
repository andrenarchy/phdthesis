from matplotlib import rc, pyplot
rc('text', usetex=True)

# use the default color cycle
colors = pyplot.rcParams['axes.color_cycle']


def complement(A, B):
    return list(set(A).difference(set(B)))
