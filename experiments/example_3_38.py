import numpy
from scipy.sparse import block_diag
from matplotlib import pyplot
from krypy import linsys
import defaults


# plot a convergence history
def plot_resnorms(resnorms, **kwargs):
    resnorms = list(resnorms)
    for i in range(len(resnorms)):
        resnorms[i] = numpy.max([resnorms[i], 1e-16])
    pyplot.semilogy(resnorms, **kwargs)


def run_figure_3_7a():
    # perturbation size
    epsilon = 1e-8

    # define base matrix
    B = numpy.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    m = B.shape[0]
    n = 20
    C = numpy.diag(numpy.ones(n-1), -1)
    C[0, -1] = 1
    A = block_diag((B, 1e2*C)).todense()

    # right hand side
    b = numpy.eye(m+n, 1)

    # error matrix
    F = numpy.zeros((m+n, m+n))
    F[m, 0] = epsilon

    # error vector
    f = numpy.eye(m+n, 1)
    f[m, 0] = epsilon

    # solve unperturbed linear system
    solver = linsys.Gmres(linsys.LinearSystem(A, b), tol=1e-13)
    plot_resnorms(solver.resnorms, label=r'$Ax=b$')

    # solve with perturbed matrix
    solver_mat = linsys.Gmres(linsys.LinearSystem(A+F, b), tol=1e-13)
    plot_resnorms(solver_mat.resnorms, ls='--', label=r'$(A+F)x_F=b$')

    # solve with perturbed right hand side
    solver_mat = linsys.Gmres(linsys.LinearSystem(A, b+f), tol=1e-13)
    plot_resnorms(solver_mat.resnorms, ls='-.', label=r'$Ax_f=b+f$')

    # add labels and legend
    pyplot.ylim(ymax=10)
    pyplot.legend()
    pyplot.xlabel(r'Iteration $i$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.show()


def run_figure_3_7b():
    # perturbation size
    epsilon = 1e-8

    # define base matrix
    B = numpy.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    m = B.shape[0]
    n = 20
    C = numpy.diag(numpy.ones(n-1), -1)
    A = block_diag((B, 1e2*C)).todense()
    A = A+A.T

    # right hand side
    b = numpy.eye(m+n, 1)

    # error matrix
    F = numpy.zeros((m+n, m+n))
    F[m, 0] = epsilon
    F = F+F.T

    # error vector
    f = numpy.eye(m+n, 1)
    f[m, 0] = epsilon

    # solve unperturbed linear system
    solver = linsys.Minres(linsys.LinearSystem(A, b, self_adjoint=True),
                           tol=1e-13)
    plot_resnorms(solver.resnorms, label=r'$Ax=b$')

    # solve with perturbed matrix
    solver_mat = linsys.Minres(linsys.LinearSystem(A+F, b, self_adjoint=True),
                               tol=1e-13)
    plot_resnorms(solver_mat.resnorms, ls='--', label=r'$(A+F)x_F=b$')

    # solve with perturbed right hand side
    solver_mat = linsys.Minres(linsys.LinearSystem(A, b+f, self_adjoint=True),
                               tol=1e-13)
    plot_resnorms(solver_mat.resnorms, ls='-.', label=r'$\hat{A}x_f=b+f$')

    # add labels and legend
    pyplot.ylim(ymax=10)
    pyplot.legend()
    pyplot.xlabel(r'Iteration $i$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.show()


if __name__ == '__main__':
    pyplot.ion()
    run_figure_3_7a()
    pyplot.figure()
    pyplot.ioff()
    run_figure_3_7b()
