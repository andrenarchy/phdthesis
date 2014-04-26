import pynosh_helper
import krypy
import numpy
from matplotlib import pyplot
import defaults

# do not use orthogonalization and double application for reasons of efficiency
projection_kwargs = {
    'orthogonalize': False,
    'iterations': 1
    }


class DiagRecMinres(pynosh_helper.RecyclingDiagnosisMixin,
                    krypy.recycling.RecyclingMinres):
    def solve(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs['projection_kwargs'] = projection_kwargs
        try:
            solver = super(DiagRecMinres, self).solve(*args, **kwargs)
        except krypy.utils.ConvergenceError as e:
            from warnings import warn
            warn(e.__str__())
            solver = e.solver
        return solver


class DiagRecGmres(pynosh_helper.RecyclingDiagnosisMixin,
                   krypy.recycling.RecyclingGmres):
    def solve(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs['projection_kwargs'] = projection_kwargs
        return super(DiagRecGmres, self).solve(*args, **kwargs)


def plot_resnorms(recycling_solver, title,
                  xlim=(0, 180),
                  xlabel='MINRES iteration $n$',
                  ylim=(1e-10, 10),
                  ylabel=r'$\|r_n\|/\|b\|$',
                  yyicklabels=True
                  ):
    pyplot.figure()

    # plot residual norms
    recycling_solver.diag_plot_resnorms()

    if xlim is not None:
        pyplot.xlim(xlim)
    if ylim is not None:
        pyplot.ylim(ylim)
    if xlabel is not None:
        pyplot.xlabel(xlabel)
    if ylabel is not None:
        pyplot.ylabel(ylabel)

    pyplot.title(title)
    pyplot.show(block=False)


def plot_ritz_values(recycling_solver, title):
    pyplot.figure()

    # plot ritz values
    recycling_solver.diag_plot_ritz()

    pyplot.title(title)
    pyplot.show(block=False)


def run_2d_vanilla():
    '''No deflation.'''
    numpy.random.seed(0)
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_2d.vtu',
        RecyclingSolver=DiagRecMinres,
        recycling_solver_kwargs={'explicit_residual': True}
        )
    plot_resnorms(recycling_solver, 'no recycling (2d)')
    plot_ritz_values(recycling_solver, 'Ritz values (2d)')

    # rerun without explicit residual for timings
    numpy.random.seed(0)
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_2d.vtu',
        RecyclingSolver=DiagRecMinres,
        )
    return recycling_solver


def run_2d_fixed():
    '''Always choose 12 Ritz vectors with Ritz values of smallest magnitude.'''
    numpy.random.seed(0)

    def vector_factory_generator(_):
        return krypy.recycling.factories.RitzFactorySimple(
            n_vectors=12,
            which='sm'
            )
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_2d.vtu',
        RecyclingSolver=DiagRecMinres,
        vector_factory_generator=vector_factory_generator
        )
    plot_resnorms(recycling_solver, '12 deflation vectors (2d)')

    return recycling_solver


def run_2d_approx():
    '''Automatically choose deflation vectors with approximate Krylov.'''
    numpy.random.seed(0)

    def vector_factory_generator(_):
        return krypy.recycling.factories.RitzFactory(
            subsets_generator=krypy.recycling.generators.RitzSmall(
                max_vectors=15,
                ),
            subset_evaluator=krypy.recycling.evaluators.RitzApproxKrylov(
                mode='extrapolate',
                pseudospectra=False,
                deflweight=2.
                ),
            #print_results='values'
            )
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_2d.vtu',
        RecyclingSolver=DiagRecMinres,
        vector_factory_generator=vector_factory_generator
        )
    plot_resnorms(recycling_solver, 'approximate Krylov strategy (2d)')

    return recycling_solver


def run_2d_apriori():
    '''Automatically choose deflation vectors with a priori bound.'''
    numpy.random.seed(0)

    def vector_factory_generator(_):
        return krypy.recycling.factories.RitzFactory(
            subsets_generator=krypy.recycling.generators.RitzSmall(
                max_vectors=15,
                ),
            subset_evaluator=krypy.recycling.evaluators.RitzApriori(
                Bound=krypy.utils.BoundMinres,
                deflweight=2.
                ),
            #print_results='values'
            )
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_2d.vtu',
        RecyclingSolver=DiagRecMinres,
        vector_factory_generator=vector_factory_generator
        )
    plot_resnorms(recycling_solver, 'quad bound + a priori strategy (2d)')

    return recycling_solver


def run_3d_vanilla():
    '''No deflation.'''
    numpy.random.seed(0)
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_3d.vtu',
        RecyclingSolver=DiagRecMinres,
        recycling_solver_kwargs={'maxiter': 280}
        )
    plot_resnorms(recycling_solver, 'no recycling (3d)',
                  xlim=(0, 280), ylabel=None)
    plot_ritz_values(recycling_solver, 'Ritz values (3d)')

    return recycling_solver


def run_3d_fixed():
    '''Always choose 12 Ritz vectors with Ritz values of smallest magnitude.'''
    numpy.random.seed(0)

    def vector_factory_generator(_):
        return krypy.recycling.factories.RitzFactorySimple(
            n_vectors=12,
            which='sm'
            )
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_3d.vtu',
        RecyclingSolver=DiagRecMinres,
        recycling_solver_kwargs={'maxiter': 280},
        vector_factory_generator=vector_factory_generator
        )
    plot_resnorms(recycling_solver, '12 deflation vectors (3d)',
                  xlim=(0, 280), ylabel=None)

    return recycling_solver


def run_3d_approx():
    '''Automatically choose deflation vectors with approximate Krylov.'''
    numpy.random.seed(0)

    def vector_factory_generator(_):
        return krypy.recycling.factories.RitzFactory(
            subsets_generator=krypy.recycling.generators.RitzSmall(
                max_vectors=20,
                ),
            subset_evaluator=krypy.recycling.evaluators.RitzApproxKrylov(
                mode='extrapolate',
                pseudospectra=False,
                deflweight=2.
                ),
            #print_results='values'
            )
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_3d.vtu',
        RecyclingSolver=DiagRecMinres,
        recycling_solver_kwargs={'maxiter': 280},
        vector_factory_generator=vector_factory_generator
        )
    plot_resnorms(recycling_solver, 'approximate Krylov strategy (2d)',
                  xlim=(0, 280), ylabel=None)

    return recycling_solver


def run_3d_apriori():
    '''Automatically choose deflation vectors with a priori bound.'''
    numpy.random.seed(0)

    def vector_factory_generator(_):
        return krypy.recycling.factories.RitzFactory(
            subsets_generator=krypy.recycling.generators.RitzSmall(
                max_vectors=20,
                ),
            subset_evaluator=krypy.recycling.evaluators.RitzApriori(
                Bound=krypy.utils.BoundMinres,
                deflweight=2.
                ),
            #print_results='values'
            )
    recycling_solver = pynosh_helper.run(
        'exp_pynosh_3d.vtu',
        RecyclingSolver=DiagRecMinres,
        recycling_solver_kwargs={'maxiter': 280},
        vector_factory_generator=vector_factory_generator
        )
    plot_resnorms(recycling_solver, 'quad bound + a priori strategy (2d)',
                  xlim=(0, 280), ylabel=None)

    return recycling_solver


def plot_jobs_timings(jobs, title):
    colors = pyplot.rcParams['axes.color_cycle']
    linestyles = ['-', '--', '-.', ':']
    from itertools import cycle

    # plot solve timings
    pyplot.figure()
    for job, color, linestyle in zip(jobs, cycle(colors), cycle(linestyles)):
        timings = job['diag']['timings']['solve']
        pyplot.plot(timings,
                    linestyle,
                    label=job['name'],
                    color=color
                    )
    pyplot.xlim(0, len(timings)-1)
    pyplot.xlabel('Newton step')
    pyplot.ylabel('Time in s')
    pyplot.title('solve timings '+title)
    pyplot.legend()
    pyplot.show(block=False)

    # plot factory timings
    pyplot.figure()
    for job, color, linestyle in zip(jobs, cycle(colors), cycle(linestyles)):
        timings = job['diag']['timings']['vector_factory']
        # do not plot timings that are close to timer resolution
        if numpy.max(timings) > 1e-3:
            pyplot.plot(range(1, len(timings)),
                        timings[1:],
                        linestyle,
                        label=job['name'],
                        color=color
                        )
    pyplot.xlim(0, len(timings)-1)
    pyplot.xlabel('Newton step')
    pyplot.ylabel('Time in s')
    pyplot.title('factory timings '+title)
    pyplot.legend()
    pyplot.show(block=False)


def plot_jobs_num_defl_vectors(jobs, title):
    colors = pyplot.rcParams['axes.color_cycle']
    linestyles = ['-', '--', '-.', ':']
    from itertools import cycle

    # plot number of deflation vectors
    pyplot.figure()
    for job, color, linestyle in zip(jobs, cycle(colors), cycle(linestyles)):
        num_defl_vectors = job['diag']['num_defl_vectors']
        if numpy.max(num_defl_vectors) > 0:
            pyplot.plot(range(1, len(num_defl_vectors)),
                        num_defl_vectors[1:],
                        linestyle,
                        label=job['name'],
                        color=color
                        )
    pyplot.xlim(0, len(num_defl_vectors)-1)
    pyplot.xlabel('Newton step')
    pyplot.ylabel('Deflation vectors')
    pyplot.legend()
    pyplot.show(block=False)


def run_2d():
    jobs_2d = [
        {'name': 'no recycling',
         'run': run_2d_vanilla},
        {'name': '12 vectors',
         'run': run_2d_fixed},
        {'name': 'quad/a-priori',
         'run': run_2d_apriori},
        {'name': 'approx. Krylov',
         'run': run_2d_approx},
        ]
    for job in jobs_2d:
        recycling_solver = job['run']()
        job['diag'] = recycling_solver.diag
    plot_jobs_timings(jobs_2d, '(2d)')
    plot_jobs_num_defl_vectors(jobs_2d, '(2d)')

    return jobs_2d


def run_3d():
    jobs_3d = [
        {'name': 'no deflation',
         'run': run_3d_vanilla},
        {'name': '12 vectors',
         'run': run_3d_fixed},
        {'name': 'quad/a-priori',
         'run': run_3d_apriori},
        {'name': 'approx. Krylov',
         'run': run_3d_approx},
        ]
    for job in jobs_3d:
        import gc
        gc.collect()
        recycling_solver = job['run']()
        job['diag'] = recycling_solver.diag
        del recycling_solver
    plot_jobs_timings(jobs_3d, '(3d)')
    plot_jobs_num_defl_vectors(jobs_3d, '(3d)')

    return jobs_3d

if __name__ == '__main__':
    run_2d()
    run_3d()
    pyplot.show(block=True)
