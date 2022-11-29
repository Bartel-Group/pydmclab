#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=1440
#SBATCH --error=log.e
#SBATCH --output=log.o
#SBATCH --account=cbartel
#SBATCH --job-name=Cr2Mn1O4-mp-28226-afm_1
#SBATCH --partition=msismall


ulimit -s unlimited

echo working on NEWRUN_gga-loose >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-loose

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-loose >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o

echo working on NEWRUN_gga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-loose/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "loose is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-loose/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-relax/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-loose/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-relax/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-relax

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o

echo working on NEWRUN_gga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-relax/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "relax is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-relax/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-static/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-relax/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-static/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-static

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o

echo working on NEWRUN_metagga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-static/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "static is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-static/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-relax/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/gga-static/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-relax/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-relax

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched metagga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o

echo working on NEWRUN_metagga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-relax/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "relax is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-relax/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-static/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-relax/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-static/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/metagga-static

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched metagga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp_standard/calcs/Cr2Mn1O4/mp-28226/afm_1/status.o
