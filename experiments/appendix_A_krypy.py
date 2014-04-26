from numpy import diag, linspace, ones, eye
from krypy.linsys import LinearSystem, Minres

# construct the linear system
A = diag(linspace(1, 2, 20))
A[0, 0] = -1e-5
b = ones(20)
linear_system = LinearSystem(A, b, self_adjoint=True)

# solve the linear system (approximate solution is solver.xk)
solver = Minres(linear_system)


from krypy.deflation import DeflatedMinres
dsolver = DeflatedMinres(linear_system, U=eye(20, 1))


from krypy.recycling import RecyclingMinres

# get recycling solver with approximate Krylov subspace strategy
rminres = RecyclingMinres(vector_factory='RitzApproxKrylov')

# solve twice
rsolver1 = rminres.solve(linear_system)
rsolver2 = rminres.solve(linear_system)


from matplotlib.pyplot import semilogy, show, legend, xlabel, ylabel
semilogy(solver.resnorms, label='original')
semilogy(dsolver.resnorms, label='exact deflation', ls='dotted')
semilogy(rsolver2.resnorms, label='automatic recycling', ls='dashed')
legend()

xlabel(r'MINRES iteration $n$')
ylabel(r'$\frac{\|r_n\|}{\|b\|}$')
show()
