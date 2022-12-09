#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=1440
#SBATCH --error=log.e
#SBATCH --output=log.o
#SBATCH --account=cbartel
#SBATCH --job-name=Cr2Mn1O4-mp-28226-nm
#SBATCH --partition=msismall


ulimit -s unlimited

echo working on NEWRUN_gga-loose >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-loose

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-loose >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o

echo working on NEWRUN_gga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-loose/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "loose is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-loose/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-relax/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-loose/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-relax/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-relax

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o

echo working on NEWRUN_gga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-relax/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "relax is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-relax/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-static/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-relax/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-static/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-static

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched gga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o

echo working on NEWRUN_metagga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-static/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "static is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-static/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-relax/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/gga-static/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-relax/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-relax

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched metagga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o

echo working on NEWRUN_metagga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
isInFile=$(cat /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-relax/OUTCAR | grep -c Elaps)
if [ $isInFile -eq 0 ]; then
   echo "relax is not done yet so this job is being killed" >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
   scancel $SLURM_JOB_ID
fi
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-relax/WAVECAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-static/WAVECAR
cp /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-relax/CONTCAR /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-static/POSCAR
cd /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/metagga-static

srun --ntasks=16 --mpi=pmi2 /home/cbartel/shared/bin/vasp/vasp_std > vasp.o

echo launched metagga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/pydmc/examples/vasp_mp/calcs/Cr2Mn1O4/mp-28226/nm/status.o
