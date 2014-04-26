import numpy
from matplotlib import pyplot
from krypy import linsys, deflation
from example_3_23 import get_problem
import defaults


# approx Krylov bound
def plot_approx(k, G_norm=0., g_norm=0.):
    problem = get_problem()
    linear_system = problem['linear_system']
    A = problem['A']
    b = problem['b']
    tol = 1e-6
    numpy.random.seed(0)

    # solve without deflation
    solver = deflation.DeflatedMinres(linear_system, tol=tol,
                                      store_arnoldi=True)
    arnoldifyer = deflation.Arnoldifyer(solver)
    ritz = deflation.Ritz(solver)
    from itertools import cycle

    G = numpy.random.rand(*A.shape)
    G += G.T.conj()
    G *= G_norm/numpy.linalg.norm(G, 2)

    g = numpy.random.rand(A.shape[0], 1)
    g *= g_norm/numpy.linalg.norm(g, 2)
    ls_pert = linsys.LinearSystem(A+G, b+g, self_adjoint=True)
    for i, color in zip(range(0, k), cycle(defaults.colors)):
        # actual convergence
        solver = deflation.DeflatedGmres(
            ls_pert, U=ritz.get_vectors(list(range(i))), tol=tol
            )
        pyplot.semilogy(solver.resnorms,
                        color=color,
                        alpha=0.3
                        )

        # compute bound
        bound_approx = deflation.bound_pseudo(arnoldifyer,
                                              ritz.coeffs[:, :i],
                                              G_norm=G_norm,
                                              GW_norm=G_norm,
                                              WGW_norm=G_norm,
                                              g_norm=g_norm
                                              )
        pyplot.semilogy(bound_approx, color=color, linestyle='dashed')


def run_figure_3_12a():
    # first 3 vectors
    plot_approx(4)
    pyplot.xlabel(r'MINRES iteration $n$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.ylim(1e-6, 2)
    pyplot.show()


def run_figure_3_12b():
    # first 10 vectors
    plot_approx(10)
    pyplot.xlabel(r'MINRES iteration $n$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.ylim(1e-15, 2)
    pyplot.show()


def run_figure_3_12c():
    # first 3 vectors with perturbations
    plot_approx(4, G_norm=1e-7, g_norm=0.1)
    pyplot.xlabel(r'MINRES iteration $n$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.ylim(1e-6, 2)
    pyplot.show()


if __name__ == '__main__':
    pyplot.ion()
    run_figure_3_12a()
    pyplot.figure()
    run_figure_3_12b()
    pyplot.ioff()
    pyplot.figure()
    run_figure_3_12c()
