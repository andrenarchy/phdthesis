import numpy
import voropy
import pynosh
import krypy
from matplotlib import pyplot


def run(filename,
        RecyclingSolver=krypy.recycling.RecyclingGmres,
        vector_factory_generator=None,
        recycling_solver_kwargs=None
        ):
    mesh, point_data, field_data = voropy.reader.read(filename)
    num_nodes = len(mesh.node_coords)

    # build the model evaluator
    mu = field_data['mu'][0]
    g = field_data['g'][0]
    V = point_data['V']

    nls_modeleval = pynosh.modelevaluator_nls.NlsModelEvaluator(
        mesh,
        V=V,
        A=point_data['A'],
        preconditioner_type='cycles',
        num_amg_cycles=1
        )

    psi0 = numpy.reshape(point_data['psi0'][:, 0]
                         + 1j * point_data['psi0'][:, 1],
                         (num_nodes, 1)
                         )

    if recycling_solver_kwargs is None:
        recycling_solver_kwargs = {}
    recycling_solver_kwargs = dict(recycling_solver_kwargs)
    if 'maxiter' not in recycling_solver_kwargs:
        recycling_solver_kwargs['maxiter'] = 200

    newton_out = pynosh.numerical_methods.newton(
        psi0,
        nls_modeleval,
        RecyclingSolver=RecyclingSolver,
        recycling_solver_kwargs=recycling_solver_kwargs,
        vector_factory_generator=vector_factory_generator,
        nonlinear_tol=1e-10,
        eta0=1e-10,
        forcing_term='constant',
        compute_f_extra_args={'g': g, 'mu': mu},
        newton_maxiter=30
        )

    sol = newton_out['x'][0:num_nodes]

    # energy of the state
    print('# Energy of the final state: %g.' % nls_modeleval.energy(sol))

    return newton_out['recycling_solver']


def serialize_dict(d, indent=0, first=False):
    indstr = indent*' '

    def serialize_item(k, v, indent=0):
        if v is None:
            return '{0}'.format(k)
        if type(v) is dict:
            return '{0} = {{\n{1}\n{2}}}'.format(
                k,
                serialize_dict(v, indent=indent+2),
                indstr
                )
        return '{0} = {1}'.format(k, v)
    pre = '\n' if first else ''
    return pre + ',\n'.join([indstr + serialize_item(k, v, indent)
                             for k, v in d.items()])


def symlog2tikz(axis,
                neg_axis_extra_opts=None,
                pos_axis_extra_opts=None,
                gplot_extra_opts=None):
    import matplotlib as mpl
    import matplotlib2tikz as mpl2t

    root_template = r'''
  {colors}
\begin{{tikzpicture}}
  \begin{{groupplot}}[{gplot_opts}]
    {gplots}
  \end{{groupplot}}
  \draw [dotted] (group c1r1.south east) -- (group c2r1.south west);
  \draw [dotted] (group c1r1.north east) -- (group c2r1.north west);
  \node[anchor=north] at ($({{group c1r1.west}}|-{{group c1r1.outer south}})!0.5!({{group c2r1.east}}|-{{group c1r1.outer south}})$) {{{xlabel}}};
\end{{tikzpicture}}'''

    plot_template = r'''
    \addplot[{0}]
    coordinates{{{1}}};'''

    negative_plots = ''
    positive_plots = ''
    mpl2t_colors_data = {
        'custom colors': {}
        }
    xabs_mins = []
    xabs_maxs = []
    ymins = []
    ymaxs = []
    for child in axis.get_children():
        if isinstance(child, mpl.lines.Line2D):
            data = child.get_data()
            xpositive = data[0] > 0
            xnegative = data[0] < 0
            if numpy.sum(xpositive + xnegative) != len(data[0]):
                from warnings import warn
                warn('zero data is invalid with symlog scale and is omitted')

            _, xcol, _ = mpl2t._mpl_color2xcolor(mpl2t_colors_data,
                                                 child.get_color())
            plot_opts = {
                xcol: None,
                'mark': '*',
                'mark size': '1',
                'only marks': None
                }
            if numpy.sum(xnegative) > 0:
                negative_plots += plot_template.format(
                    serialize_dict(plot_opts, 6, first=True),
                    ' '.join(['({0},{1})'.format(-x, y)
                              for x, y in zip(data[0][xnegative],
                                              data[1][xnegative])
                              ])
                    )
            if numpy.sum(xpositive) > 0:
                positive_plots += plot_template.format(
                    serialize_dict(plot_opts, 6, first=True),
                    ' '.join(['({0},{1})'.format(x, y)
                              for x, y in zip(data[0][xpositive],
                                              data[1][xpositive])
                              ])
                    )
            xvals = data[0][xnegative+xpositive]
            xabs_mins.append(numpy.min(numpy.abs(xvals)))
            xabs_maxs.append(numpy.max(numpy.abs(xvals)))
            yvals = data[1][xnegative+xpositive]
            ymins.append(numpy.min(yvals))
            ymaxs.append(numpy.max(yvals))

    gplot_opts = {
        'group style': {
            'group size': '2 by 1',
            'horizontal sep': '3em',
            'every plot/.style': {
                'scale only axis': None,
                'ymin': min(ymins),
                'ymax': max(ymaxs),
                'enlarge y limits': '0.025'
                }
            }
        }
    if gplot_extra_opts is not None:
        gplot_opts.update(gplot_extra_opts)

    nextgplot_opts = {
        'height': r'0.5*\textwidth',
        'width': r'0.4*\textwidth',
        'xmode': 'log',
        'enlarge x limits': '0.1',
        'xmin': min(xabs_mins),
        'xmax': max(xabs_maxs)
        }

    nextgplot_template = r'''
    \nextgroupplot[{0}]'''

    neg_nextgplot_opts = {
        'axis y line*': 'left',
        'x dir': 'reverse',
        'xticklabel': r'-\axisdefaultticklabellog',
        'ylabel': axis.get_ylabel()
        }
    neg_nextgplot_opts.update(nextgplot_opts)
    if neg_axis_extra_opts is not None:
        neg_nextgplot_opts.update(neg_axis_extra_opts)

    pos_nextgplot_opts = {
        'axis y line*': 'right',
        'yticklabel': r'\empty'
        }
    pos_nextgplot_opts.update(nextgplot_opts)
    if pos_axis_extra_opts is not None:
        pos_nextgplot_opts.update(pos_axis_extra_opts)

    return root_template.format(
        colors='\n'.join(mpl2t._get_color_definitions(mpl2t_colors_data)),
        gplot_opts=serialize_dict(gplot_opts, 4, first=True),
        gplots=nextgplot_template.format(
            serialize_dict(neg_nextgplot_opts, 6, first=True)
            )
        + negative_plots
        + nextgplot_template.format(
            serialize_dict(pos_nextgplot_opts, 6, first=True)
            )
        + positive_plots,
        xlabel=axis.get_xlabel()
        )


class RecyclingDiagnosisMixin(object):
    def __init__(self, *args, **kwargs):
        self.diag = {
            'i': 0,
            'resnorms': [],
            'ritz_values': [],
            'num_defl_vectors': []
            }

        super(RecyclingDiagnosisMixin, self).__init__(*args, **kwargs)
        self.diag['timings'] = self.timings

    def solve(self, *args, **kwargs):
        print('solving linear system {0}... '.format(self.diag['i'])),

        # time solve()
        err = None
        try:
            solver = super(RecyclingDiagnosisMixin, self).solve(*args,
                                                                **kwargs)
        except krypy.utils.ConvergenceError as e:
            solver = e.solver
            err = e

        # append residual norms
        self.diag['resnorms'].append(list(solver.resnorms))

        # compute Ritz values
        ritz = krypy.deflation.Ritz(solver)
        self.diag['ritz_values'].append(list(ritz.values))

        self.diag['num_defl_vectors'].append(solver.projection.U.shape[1])

        # bump i
        self.diag['i'] += 1

        # print basic diag output
        print('{:.5f}s/{:.5f}s (vector factory/solve), {:} defl vecs.'
              .format(self.diag['timings']['vector_factory'][-1],
                      self.diag['timings']['solve'][-1],
                      self.diag['num_defl_vectors'][-1]
                      )
              )

        if err is not None:
            raise err

        return solver

    def diag_plot_resnorms(self, color=None):
        newton_steps = len(self.diag['resnorms'])
        if color is None:
            color = pyplot.rcParams['axes.color_cycle'][0]
        for index, resnorms in enumerate(self.diag['resnorms']):
            pyplot.semilogy(
                resnorms,
                color=color,
                alpha=(0.25+0.75*float(index)/newton_steps)
                )

    def diag_plot_ritz(self):
        mins = []
        for index, ritz_values in enumerate(self.diag['ritz_values']):
            pyplot.plot(ritz_values,
                        len(ritz_values)*[index+1],
                        '.')
            mins.append(numpy.min(numpy.abs(ritz_values)))
        pyplot.xscale('symlog', linthreshx=numpy.min(mins))
        pyplot.xlabel(r'Ritz values of $M_iJ_i$')
        pyplot.ylabel(r'Newton step $i$')
