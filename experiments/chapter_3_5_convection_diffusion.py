import dolfin
from dolfin import dx, inner, nabla_grad, near
import krypy
import numpy
from matplotlib import pyplot
import defaults

dolfin.parameters.linear_algebra_backend = "uBLAS"


def mat_dolfin2sparse(A):
    rows, cols, values = A.data()
    from scipy.sparse import csr_matrix
    return csr_matrix((values, cols, rows))


# element stabilization term for SUPG (streamline diffusion)
class Stabilization(dolfin.Expression):
    def __init__(self, mesh, wind, epsilon):
        self.mesh = mesh
        self.wind = wind
        self.epsilon = epsilon

    def eval_cell(self, values, x, ufc_cell):
        wk, hk = self.get_wind(ufc_cell)

        # element Peclet number
        Pk = wk*hk/(2*self.epsilon)

        if Pk > 1:
            values[:] = hk/(2*wk)*(1-1/Pk)
        else:
            values[:] = 0.

    def get_wind(self, ufc_cell):
        '''|w_k| and h_k as in ElmSW05'''
        cell = dolfin.Cell(self.mesh, ufc_cell.index)

        # compute centroid
        dim = self.mesh.topology().dim()
        centroid = numpy.zeros(dim)
        vertices = cell.get_vertex_coordinates()
        for i in range(dim):
            centroid[i] = numpy.sum(vertices[i::dim])
        centroid /= (vertices.shape[0]/dim)

        # evaluate wind and its norm |w_k|
        wind = numpy.array([self.wind[i](centroid) for i in range(dim)])
        wk = numpy.linalg.norm(wind, 2)

        # compute element length in direction of wind
        # TODO: this has to be tweaked for general wind vectors
        hk = cell.diameter()

        return wk, hk


def get_ilu(A, fill_factor=1):
    # setup preconditioner
    from scipy.sparse.linalg import spilu
    B_ilu = spilu(A, fill_factor=fill_factor, permc_spec='MMD_AT_PLUS_A')

    def _apply_ilu(x):
        ret = numpy.zeros(x.shape, dtype=x.dtype)
        for i in range(x.shape[1]):
            if numpy.iscomplexobj(x):
                ret[:, i] = (B_ilu.solve(numpy.real(x[:, i]))
                             + 1j*B_ilu.solve(numpy.imag(x[:, i])))
            else:
                ret[:, i] = B_ilu.solve(x[:, i])
        return ret
    B = krypy.utils.LinearOperator(A.shape, dtype=A.dtype,
                                   dot=_apply_ilu)
    return B, B_ilu


def get_conv_diff_ls(mesh, V, wind, right_boundary,
                     f=None,
                     timed=False,
                     return_mat=False
                     ):
    # right hand side
    if f is None:
        f = dolfin.Constant(0.)

    # diffusivity
    epsilon = 1./200

    # convection field
    delta = Stabilization(mesh, wind, epsilon)

    # define boundary conditions
    class Boundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class BoundaryRight(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1.)
    boundaries = dolfin.FacetFunction('size_t', mesh)
    boundaries.set_all(0)
    boundary = Boundary()
    boundary.mark(boundaries, 1)
    boundary2 = BoundaryRight()
    boundary2.mark(boundaries, 2)
    boundary
    bcs = [dolfin.DirichletBC(V, dolfin.Constant(0.), boundaries, 1),
           dolfin.DirichletBC(V, right_boundary, boundaries, 2)
           ]

    # variational formulation
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    def get_conv_diff(u, v, epsilon, wind, stabilize=True):
        a = (
            epsilon*inner(nabla_grad(u), nabla_grad(v))*dx
            + inner(nabla_grad(u), wind)*v*dx
            )
        L = f*v*dx
        if stabilize:
            a += delta*inner(wind, nabla_grad(u))*inner(wind, nabla_grad(v))*dx
            L += delta*f*inner(wind, nabla_grad(v))*dx
        return a, L

    a, L = get_conv_diff(u, v, epsilon, wind)

    A = dolfin.assemble(a)
    b = dolfin.assemble(L)
    u0 = dolfin.Function(V).vector()
    u0.zero()

    [bc.apply(A, b) for bc in bcs]
    [bc.apply(u0) for bc in bcs]

    import scipy.sparse
    A_mat = scipy.sparse.csc_matrix(mat_dolfin2sparse(A))

    return A_mat, b.array(), u0.array()


def run_figure_3_15():
    # mesh and function space
    mesh = dolfin.RectangleMesh(-1, -1, 1, 1, 25, 25, 'crossed')
    V = dolfin.FunctionSpace(mesh, 'Lagrange', 1)

    wind = dolfin.Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'))
    boundary = dolfin.Constant(1.)

    # get matric with dolfin
    A, b, x0 = get_conv_diff_ls(
        mesh, V, wind,
        boundary,
        timed=True,
        return_mat=True
        )

    # get preconditioner
    Ml, Ml_ilu = get_ilu(A, fill_factor=2)

    # get linear system
    ls = krypy.linsys.TimedLinearSystem(A, b, Ml=Ml)

    # set parameters for solvers
    tol = 1e-12
    maxiter = 200

    # solve without deflation
    pyplot.figure()
    orig_solver = krypy.deflation.DeflatedGmres(ls,
                                                x0=x0,
                                                store_arnoldi=True,
                                                tol=tol,
                                                maxiter=maxiter
                                                )

    pyplot.semilogy(orig_solver.resnorms, alpha=0.3)

    # compute Ritz pairs and get arnoldifyer
    ritz = krypy.deflation.Ritz(orig_solver)
    sort = numpy.argsort(numpy.abs(ritz.resnorms))
    arnoldifyer = krypy.deflation.Arnoldifyer(orig_solver)

    # solve the same linear system with deflation
    from itertools import cycle
    for n_defl_vecs, color in zip(range(16), cycle(defaults.colors)):
        print('considering {0} deflation vectors'.format(n_defl_vecs))

        # solve with deflation
        solver = krypy.deflation.DeflatedGmres(
            ls,
            x0=x0,
            U=ritz.get_vectors(sort[:n_defl_vecs]),
            tol=tol,
            maxiter=maxiter
            )
        if n_defl_vecs == 0:
            alpha = 1.0
        else:
            alpha = 0.3
        pyplot.semilogy(solver.resnorms, alpha=alpha, color=color)

        if n_defl_vecs > 0:
            # compute full-flavoured bound
            resnorms = krypy.deflation.bound_pseudo(
                arnoldifyer,
                ritz.coeffs[:, sort[:n_defl_vecs]],
                tol=tol,
                pseudo_type='auto',
                pseudo_kwargs={
                    'n_circles': 10,
                    'n_points': 4
                    }
                )
            pyplot.semilogy(resnorms, ls='dashed', color=color)

            # compute stripped-down bound
            resnorms = krypy.deflation.bound_pseudo(
                arnoldifyer,
                ritz.coeffs[:, sort[:n_defl_vecs]],
                tol=tol,
                pseudo_type='omit',
                )
            resnorms[-1] = 1e-15
            pyplot.semilogy(resnorms, ls='dotted', color=color)

    pyplot.ylim(1e-12, 2)
    pyplot.xlabel(r'GMRES iteration $n$')
    pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
    pyplot.show(block=False)

    # solve linear system with perturbed right hand side
    for num_system, gamma in enumerate([1e-3, 1e-2]):
        pyplot.figure()
        print('gamma={0}'.format(gamma))

        # get matrix with dolfin
        A_2, b_2, x0_2 = get_conv_diff_ls(
            mesh, V, wind,
            dolfin.Expression('1.+gamma*(1-x[1]*x[1])', gamma=gamma),
            timed=True,
            return_mat=True
            )
        # get linear system
        ls_2 = krypy.linsys.TimedLinearSystem(A_2, b_2, Ml=Ml)

        g_norm = numpy.linalg.norm(ls.MMlb - ls_2.MMlb, 2)
        print('|g| = |b-c| = {0}'.format(g_norm))

        from itertools import cycle
        for n_defl_vecs, color in zip(range(11), cycle(defaults.colors)):
            print('considering {0} deflation vectors'.format(n_defl_vecs))

            # solve with deflation
            solver = krypy.deflation.DeflatedGmres(
                ls_2,
                x0=x0_2,
                U=ritz.get_vectors(sort[:n_defl_vecs]),
                tol=tol,
                maxiter=maxiter
                )
            pyplot.semilogy(solver.resnorms, alpha=0.3, color=color)

            # compute full-flavoured bound
            resnorms = krypy.deflation.bound_pseudo(
                arnoldifyer,
                ritz.coeffs[:, sort[:n_defl_vecs]],
                tol=tol,
                g_norm=g_norm,
                pseudo_type='auto',
                pseudo_kwargs={
                    'n_circles': 10,
                    'n_points': 4
                    },
                terminate_factor=1.5
                )
            pyplot.semilogy(resnorms, ls='dashed', color=color)

        pyplot.ylim(1e-12, 2)
        pyplot.xlabel(r'GMRES iteration $n$')
        pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
        pyplot.show(block=False)

    # solve linear system with perturbed matrix
    for num_system, gamma in enumerate([1e-6, 1e-5]):
        pyplot.figure()
        print('gamma={0}'.format(gamma))

        # get matric with dolfin
        A_2, b_2, x0_2 = get_conv_diff_ls(
            mesh, V, dolfin.Constant(1+gamma)*wind,
            boundary,
            timed=True,
            return_mat=True
            )
        # get linear system
        ls_2 = krypy.linsys.TimedLinearSystem(A_2, b_2, Ml=Ml)

        MlA = Ml * numpy.array(A.todense())
        MlA_2 = Ml * numpy.array(A_2.todense())
        G_norm = numpy.linalg.norm(MlA - MlA_2, 2)
        print('|G| = |Ml*A - Ml*A_2| = {0}'.format(G_norm))

        from itertools import cycle
        for n_defl_vecs, color in zip(range(11), cycle(defaults.colors)):
            print('considering {0} deflation vectors'.format(n_defl_vecs))

            # solve with deflation
            solver = krypy.deflation.DeflatedGmres(
                ls_2,
                x0=x0_2,
                U=ritz.get_vectors(sort[:n_defl_vecs]),
                tol=tol,
                maxiter=maxiter
                )
            pyplot.semilogy(solver.resnorms, alpha=0.3, color=color)

            # compute full-flavoured bound
            resnorms = krypy.deflation.bound_pseudo(
                arnoldifyer,
                ritz.coeffs[:, sort[:n_defl_vecs]],
                tol=tol,
                G_norm=G_norm,
                GW_norm=G_norm,
                WGW_norm=G_norm,
                pseudo_type='auto',
                pseudo_kwargs={
                    'n_circles': 10,
                    'n_points': 4
                    },
                terminate_factor=100
                )
            pyplot.semilogy(resnorms, ls='dashed', color=color)

        pyplot.ylim(1e-12, 2)
        pyplot.xlabel(r'GMRES iteration $n$')
        pyplot.ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
        pyplot.show(block=False)


if __name__ == '__main__':
    run_figure_3_15()
    pyplot.show(block=True)
