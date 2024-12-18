# MFC CDT Numerics Course

Top level scope contains the folders for the 1st and 2nd assignments, as well as some miscellaneous files.
There are internal READMEs within each folder

## Part 1: Finite Difference Methods in 1 space dimension

Lecturer: Dr. Hillary Weller ([h.weller@reading.ac.uk.](h.weller@reading.ac.uk.))

The first half of the course served as an introduction to numerical solutions of PDEs by focusing entirely on finite difference methods. This included forward, backward, and centred differencing schemes (implicit and explicit methods) as well as convergence analyses of them. It also contained an introduction to von Neumann stability analysis such that analytical results for the CFL condition and dispersion relations can be derived.

For the assigment, I chose to study the non-linear Shallow Water Equations using an FTBS/FS approach which included the concept of characterstic variables for improved stability. See the `Numerics-assignment` folder for more details.

## Part 2: Finite Element Methods in 2 space dimensions

Lecturer: Prof. Ian Hawke ([i.hawke@soton.ac.uk](i.hawke@soton.ac.uk))

The second half of the course gave us an introduction to the far more complex, but equally more versitle method of finite elements. This began with an outline of the method in one space dimension such that we could get to grips with the algorithm. (This also included some exercises which are seen here in `Numerics_part_2_chpt3exercises.py`.) The course then moved on to discussing the algorithm in two space dimensions and ultimately, including time dependence as well. There was also a note on discontinous Galerkin methods in the final section, although this proved to advanced a concept to be implemented in the assignment.

For the assignment, given the task of modelling the spread of pollution after the 2005 fire at Southampton University, I elected to write a static advection-diffusion FE solver, before adding time dependence only when I was confident that the static solver worked. See the `Numerics-assignment-2` folder for more details.