import numpy
from matplotlib import pyplot
from krypy import deflation
from example_3_23 import get_problem
import defaults


def run_figure_3_11():
    problem = get_problem()
    linear_system = problem['linear_system']
    tol = 1e-6

    numpy.random.seed(0)

    minres = deflation.DeflatedMinres(linear_system, tol=tol,
                                      store_arnoldi=True)
    ritz = deflation.Ritz(minres)

    sort = numpy.argsort(ritz.values)
    ritz_coeffs = ritz.coeffs[:, sort]

    # Arnoldify
    arnoldifyer = deflation.Arnoldifyer(minres)
    Wt = ritz_coeffs[:, :3]

    Hh, Rh, q_norm, bdiff_norm, PWAW_norm, Vh, F = arnoldifyer.get(
        Wt, full=True)

    small = numpy.argwhere(numpy.abs(numpy.diag(Hh[1:, :-1]))
                           < 1e-14*numpy.linalg.norm(Hh, 2))
    l = Hh.shape[0] if small.size == 0 else numpy.min(small)

    # compute perturbation norms
    Rh_norms = [numpy.linalg.norm(Rh[:, :i], 2) for i in range(1, l+1)]

    # plot perturbation norms
    pyplot.semilogy(range(1, l+1), Rh_norms)
    pyplot.xlabel(r'Iteration $i$')
    pyplot.ylabel(r'$\|F_i\|$')
    pyplot.show()


if __name__ == '__main__':
    run_figure_3_11()
