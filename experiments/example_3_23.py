import numpy
from matplotlib import pyplot
from krypy import linsys, utils, deflation


def get_problem():
    '''Returns the problem as a dictionary.'''
    problem = {}

    evals = problem['evals'] = numpy.array(
        [-1e-3, -1e-4, -1e-5] + list(1+numpy.linspace(0, 1, 101))
        )
    N = problem['N'] = len(evals)
    A = problem['A'] = numpy.diag(evals)
    b = problem['b'] = numpy.ones((N, 1))
    b[3:] *= 1e-1
    problem['linear_system'] = linsys.LinearSystem(
        A, b, self_adjoint=True
        )

    return problem


def run_figure_3_2():
    '''Run figure 3.2.'''
    problem = get_problem()
    linear_system = problem['linear_system']
    N = problem['N']
    tol = 1e-6

    # create perturbation
    numpy.random.seed(0)
    E = numpy.random.rand(N, 3)
    E /= utils.norm(E)
    pert = 1e-5

    # use the default color cycle
    colors = pyplot.rcParams['axes.color_cycle']

    # solve vanilla
    minres = linsys.Minres(linear_system, tol=tol, store_arnoldi=True)
    pyplot.semilogy(minres.resnorms, color=colors[0], label=r'$Ax=b$')

    # plot bound for vanilla
    bound = utils.BoundMinres(problem['evals'])
    pyplot.semilogy([bound.eval_step(step)
                     for step in range(len(minres.resnorms))],
                    color=colors[0], linestyle='dashed',
                    label=r'bound for $Ax=b$'
                    )

    # solve deflated exact invariant
    V = numpy.eye(N, N)[:, :3]
    minres_V = deflation.DeflatedMinres(linear_system, U=V, tol=tol)
    pyplot.semilogy(minres_V.resnorms, color=colors[1],
                    label=r'$P_{V^\perp}Ax=P_{V^\perp}b$'
                    )

    # plot bound for exact deflation
    bound = utils.BoundMinres(problem['evals'][3:])
    pyplot.semilogy([bound.eval_step(step)
                     for step in range(len(minres_V.resnorms))],
                    color=colors[1], linestyle='dashed',
                    label=r'bound for $P_{V^\perp}Ax=P_{V^\perp}b$'
                    )

    # solve deflated approximate invariant
    minres_U = deflation.DeflatedMinres(linear_system, U=V+pert*E, tol=tol)
    pyplot.semilogy(minres_U.resnorms, '-.', color=colors[2],
                    label=r'$P_{U^\perp,AU}Ax=P_{U^\perp,AU}b$'
                    )

    # add labels and legend
    pyplot.xlabel(r'MINRES iteration $n$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.ylim(1e-6, 3)
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    run_figure_3_2()
