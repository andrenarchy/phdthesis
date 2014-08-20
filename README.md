### Recycling Krylov subspace methods: analysis and applications (PhD thesis)

This repository contains the [PhD thesis *Recycling Krylov subspace methods: analysis and applications*](http://opus4.kobv.de/opus4-tuberlin/frontdoor/index/index/docId/5576) written by André Gaul and provides the source code for all experiments as well as for the thesis itself.

#### PhD thesis (pdf and LaTeX source code)
The thesis is available as a [pdf](http://opus4.kobv.de/opus4-tuberlin/files/5576/gaul_andre.pdf). Its LaTeX source code can be found in the directory `tex-src`.

**License:** the [pdf](http://opus4.kobv.de/opus4-tuberlin/files/5576/gaul_andre.pdf)  and the LaTeX source code in the directory `tex-src` of the PhD thesis are licensed under the [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/deed.en_US).

#### Experiments (Python source code)
All experiments in the thesis can be reproduced easily with the Python source code in the directory `experiments` of this repository. The experiments require the following Python packages:
 * [KryPy](https://github.com/andrenarchy/krypy) (>=2.1.1)
 * [PseudoPy](https://github.com/andrenarchy/pseudopy) (>=1.2.1)
 * [PyNosh](https://github.com/nschloe/pynosh) (>=0.1.2)
 * matplotlib

**Data files:** the meshes and initial data for the experiments with [PyNosh](https://github.com/nschloe/pynosh) can be found here:
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.11074.png)](http://dx.doi.org/10.5281/zenodo.11074). The files should be placed in the directory `experiments`.

**License:** the files in the directory `experiments` are licensed under the [MIT license](http://opensource.org/licenses/mit-license.php).
