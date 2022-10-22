#!/bin/bash#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --walltime=86400
#SBATCH --mem=None
#SBATCH --error=log.e
#SBATCH --output=log.o
#SBATCH --constraint=None
#SBATCH --qos=None
#SBATCH --account=cbartel
#SBATCH --job-name=my-job



Working on NEWRUN_gga-loose >> ../dev/LiMn2O4/mp-22584/fm/status.o
cd ../dev/LiMn2O4/mp-22584/fm/gga-loose

mpirun -n 24 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-loose >> ../dev/LiMn2O4/mp-22584/fm/status.o

Working on NEWRUN_gga-relax >> ../dev/LiMn2O4/mp-22584/fm/status.o
cp ../dev/LiMn2O4/mp-22584/fm/gga-loose/WAVECAR ../dev/LiMn2O4/mp-22584/fm/gga-relax/WAVECAR
cp ../dev/LiMn2O4/mp-22584/fm/gga-loose/CONTCAR ../dev/LiMn2O4/mp-22584/fm/gga-relax/CONTCAR
cd ../dev/LiMn2O4/mp-22584/fm/gga-relax

mpirun -n 24 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-relax >> ../dev/LiMn2O4/mp-22584/fm/status.o

Working on NEWRUN_gga-static >> ../dev/LiMn2O4/mp-22584/fm/status.o
cp ../dev/LiMn2O4/mp-22584/fm/gga-relax/WAVECAR ../dev/LiMn2O4/mp-22584/fm/gga-static/WAVECAR
cp ../dev/LiMn2O4/mp-22584/fm/gga-relax/CONTCAR ../dev/LiMn2O4/mp-22584/fm/gga-static/CONTCAR
cd ../dev/LiMn2O4/mp-22584/fm/gga-static

mpirun -n 24 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-static >> ../dev/LiMn2O4/mp-22584/fm/status.o
