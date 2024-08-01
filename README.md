# BasicSWEx
Simplified SWE solver using FEniCSx

This should work with conda installation of FEniCSx:

https://github.com/FEniCS/dolfinx

Basic tutorial for FEniCSx is at:

https://jsdokken.com/dolfinx-tutorial/index.html


## Installation Notes

It appears there was a big update and so some minor changes need to be made to comply with fenicsx updates

For now the quickest way to get this test running is to install dolfin-fenicsx version 0.6.0 via conda:

First, install conda and then on the command line run:

conda create -n fenicsx-env

conda activate fenicsx-env

conda install -c conda-forge fenics-dolfinx mpich pyvista

Once this is complete, the test case can be ran with:

python3 main.py

this should generate some Paraview output as well as a few .csv files and .png depicting time series of point output


For HDG component,  090 is needed but is not available on conda, we can use docker for now

docker pull dolfinx/dolfinx:nightly
docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 dolfinx/dolfinx:nightly