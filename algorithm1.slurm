#!/bin/bash
#SBATCH --job-name=alg1          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G is default)
#SBATCH --array=0-9
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=             # put user email here

python algorithm1_1.py --array-task-id $SLURM_ARRAY_TASK_ID --num-tasks 10
python algorithm1_2.py --array-task-id $SLURM_ARRAY_TASK_ID --num-tasks 10
python algorithm1_3.py --array-task-id $SLURM_ARRAY_TASK_ID --num-tasks 10
python algorithm1_4.py --array-task-id $SLURM_ARRAY_TASK_ID --num-tasks 10
