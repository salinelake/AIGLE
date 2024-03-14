# AIGLE
Supplemental materials for "Ab initio generalized Langevin equations" (arxiv.org/abs/2211.06558)

This is a proof-of-concept work. The code is not ready for any productive use. Future development of coarse-grained lattice dynamics will be carried out in another software  package. 

The folder DomainMotion1D contains the implementation of one dimensional AIGLE, and a Deep Potential model for PbTiO3. DomainMotion1D/demo.ipynb can reproduce the results in Sec.III(A) of the paper.

The folder LatticeDynamics contains the implementation of multi-dimensional AIGLE for lattice dynamics. LatticeDynamics/demo.ipynb shows the workflow for training AIGLE from moelcular dynamics trajectories of collective variables. LatticeDynamics/domain_relaxation.ipynb demonstrates the application of AIGLE for simulating domain dynamics (see Fig.3 of the paper). The data files of molecular dynamics trajectories are not uploaded due to their large sizes. So currently you are not able to run these jupyter notebooks directly. But the results from a local run is kept in the notebooks.


