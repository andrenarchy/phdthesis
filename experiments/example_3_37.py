import numpy
from matplotlib import pyplot
from krypy import utils
from example_3_23 import get_problem
import defaults


def ritz(A, U):
    '''Compute Ritz values and residual norm.'''
    # compute ritz values and vectors
    X, _ = utils.qr(U)
    ritz_vals, Y = numpy.linalg.eig(X.T.conj().dot(A.dot(X)))
    ritz_vecs = X.dot(Y)

    # compute ritz residual and its norm
    R = A.dot(ritz_vecs) - ritz_vecs*ritz_vals
    Rnorm = utils.norm(R)

    return ritz_vals, Rnorm


def mathias_bound(Rnorm, delta):
    '''Compute eigenvalue bound of Mathias.'''
    if delta is None or delta == 0:
        raise utils.AssumptionError(
            'spectral gap delta does not exist or is zero.'
            )

    # does not seem to be necessary:
    #if Rnorm >= delta:
    #    raise ValueError('Rnorm = {0} >= delta = {1}'.format(Rnorm, delta))

    return Rnorm**2 / delta


def spectral_diff(A, U, evals, evals_select):
    '''Difference of spectra between exact and approximate deflation.'''
    P = utils.Projection(U, A.dot(U))
    PA = P.apply_complement(A)
    PAevals = numpy.linalg.eigvalsh(PA)
    sort = numpy.argsort(numpy.abs(PAevals))
    PAevals = numpy.sort(PAevals[sort[len(evals_select):]])
    evals = numpy.sort(
        evals[defaults.complement(range(A.shape[0]), evals_select)]
        )
    return numpy.max(numpy.abs(evals - PAevals))


def spectral_bound_angle(A, U,  evals, evals_select):
    '''Compute angle based bound.'''
    # compute angles
    thetas = utils.angles(U, A.dot(U))
    theta = numpy.max(thetas)

    # get ritz values and ritz residual norm
    ritz_vals, Rnorm = ritz(A, U)

    # determine complement eval set
    evals_comp = evals[defaults.complement(range(A.shape[0]), evals_select)]

    # get gap
    delta = utils.gap(ritz_vals, evals_comp, mode='interval')

    if delta is None or delta == 0:
        raise utils.AssumptionError(
            'spectral gap delta does not exist or is zero.'
            )

    return ((2*utils.norm(A)*Rnorm/delta
             + numpy.max(numpy.abs(evals[evals_select]))*numpy.sin(theta))
            / numpy.cos(theta))


def spectral_bound_quad(A, U, evals, evals_select):
    '''Compute quadraditc residual bound.'''
    # get ritz values and ritz residual norm
    ritz_vals, Rnorm = ritz(A, U)

    # determine complement eval set
    evals_comp = evals[defaults.complement(range(A.shape[0]), evals_select)]

    def bound(Rnorm, delta):
        return mathias_bound(Rnorm, delta) \
            + Rnorm**2/numpy.min(numpy.abs(ritz_vals))

    # bound for all eigenvalues combined
    return bound(Rnorm, utils.gap(ritz_vals, evals_comp))


def gather(fun, A, V, E, perts, evals, evals_select):
    ret_perts = []
    ret_vals = []
    for pert in perts:
        U = V + pert*E
        try:
            ret_vals.append(fun(A, U, evals, evals_select))
            ret_perts.append(pert)
        except utils.AssumptionError:
            break
    return ret_perts, ret_vals


def plot(d):
    '''Plot with deflation of d smallest eigenvalues'''
    problem = get_problem()
    A = problem['A']
    N = problem['N']
    evals = problem['evals']
    V = numpy.eye(N, d)
    sel = list(range(d))

    # get perturbations
    numpy.random.seed(0)
    E = numpy.random.rand(N, d)
    E /= utils.norm(E)
    perts = numpy.logspace(-12, 0, 300)

    pyplot.figure()

    # compute exact eigenvalue error
    diff_perts, diff_vals = gather(spectral_diff, A, V, E, perts, evals, sel)
    pyplot.loglog(diff_perts, diff_vals,
                  label=r'$\max_i|\hat{\lambda}_i - \underline{\lambda}_i|$')

    # compute angle based bound
    angle_perts, angle_vals = gather(spectral_bound_angle, A, V, E, perts,
                                     evals, sel)
    pyplot.loglog(angle_perts, angle_vals, '--', label=r'angle bound')

    # compute quadratic bound
    quad_perts, quad_vals = gather(spectral_bound_quad, A, V, E, perts,
                                   evals, sel)
    pyplot.loglog(quad_perts, quad_vals, '-.', label=r'$\|R\|^2$ bound')

    # add labels and legend
    pyplot.xlabel(r'Perturbation $\varepsilon$')
    pyplot.ylabel(r'Eigenvalue error')
    pyplot.legend(loc='upper left')
    pyplot.ylim(ymin=1e-15)


def run_figure_3_6():
    '''Run figure 3.6'''
    pyplot.ion()
    plot(3)
    plot(2)
    pyplot.show(block=True)

if __name__ == '__main__':
    run_figure_3_6()
