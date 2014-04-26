import numpy
from scipy.sparse import block_diag
from matplotlib import pyplot
from krypy import utils
import defaults


def run_figure_3_8():
    # perturbation size
    epsilon = 1e-8

    # define base matrix
    B = numpy.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    n = 20
    C = numpy.diag(numpy.ones(n-1), -1)
    C[0, -1] = 1
    A = block_diag((B, 1e2*C)).todense()

    # get pseudospectrum (normal matrix!)
    from pseudopy import Normal
    pseudo = Normal(A)

    # get polynomial
    p = utils.NormalizedRootsPolynomial(numpy.linalg.eigvals(B))

    # define deltas for evaluation
    deltas = numpy.logspace(numpy.log10(epsilon*1.01), 8, 400)

    # compute bound
    bound = utils.bound_perturbed_gmres(pseudo, p, epsilon, deltas)
    pyplot.loglog(deltas, bound)

    # add labels and legend
    pyplot.xlabel(r'$\delta$')
    pyplot.ylabel(r'bound')
    pyplot.show()


if __name__ == '__main__':
    run_figure_3_8()
