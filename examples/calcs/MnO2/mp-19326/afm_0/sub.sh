#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=1440
#SBATCH --error=log.e
#SBATCH --output=log.o
#SBATCH --account=cbartel
#SBATCH --job-name=MnO2-mp-19326-afm_0
#SBATCH --partition=msismall


ulimit -s unlimited

echo working on DONE_gga-loose >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o
echo gga-loose is done >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o

echo working on DONE_gga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o
echo gga-relax is done >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o

echo working on DONE_gga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o
echo gga-static is done >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o

echo working on DONE_metagga-relax >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o
echo metagga-relax is done >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o

echo working on DONE_metagga-static >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o
echo metagga-static is done >> /panfs/jay/groups/26/cbartel/cbartel/bin/pydmc/examples/calcs/MnO2/mp-19326/afm_0/status.o
