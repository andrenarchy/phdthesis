import numpy
from matplotlib import pyplot
from krypy import deflation
from example_3_23 import get_problem
import defaults


def run_figure_3_10():
    problem = get_problem()
    linear_system = problem['linear_system']
    tol = 1e-6

    # run plain MINRES
    minres = deflation.DeflatedMinres(linear_system, tol=tol,
                                      store_arnoldi=True)
    pyplot.semilogy(minres.resnorms, color=defaults.colors[0],
                    label=r'$Ax=b$'
                    )

    # compute Ritz pairs
    ritz = deflation.Ritz(minres)

    # run deflated MINRES
    minres_defl = deflation.DeflatedMinres(
        linear_system, U=ritz.get_vectors([0, 1, 2]), tol=tol,
        store_arnoldi=True
        )
    pyplot.semilogy(minres_defl.resnorms, color=defaults.colors[1],
                    label=r'$P_{W^\perp,AW}Ax=P_{W^\perp,AW}b$'
                    )

    # compute eval inclusion intervals
    from krypy.recycling.evaluators import RitzApriori
    intervals = RitzApriori._estimate_eval_intervals(
        ritz, [0, 1, 2], list(range(3, len(ritz.values)))
        )

    # evaluate and plot bound
    from krypy.utils import BoundMinres
    bound = BoundMinres(intervals)
    max_step = bound.get_step(1e-6)
    bound_apriori = [bound.eval_step(step)
                     for step in range(int(numpy.ceil(max_step)+1))]

    pyplot.semilogy(bound_apriori, color=defaults.colors[1], ls='dashed',
                    label=r'bound'
                    )

    pyplot.xlabel(r'MINRES iteration $n$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.ylim(1e-6, 3)
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    run_figure_3_10()
