#!/bin/bash -l

#SBATCH
#SBATCH --get-user-env
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=MOT-pre-trial-outputs
#SBATCH --time=15:00:00
#SBATCH --mail-user=supadhy6@jhu.edu
#SBATCH --mail-type=end

# Load modules
#module restore mymodules

# SET THE PYTHON PATH VARIABLE
#export PYTHONPATH="/home-4/supadhy6@jhu.edu/.conda/envs/adiLab/lib/python3.6/site-packages/"

# Job environment variables
# create these directories if not present, and save it in a convenient location
path=/data/jflomba1
JOB_DIR=$path/MOT_json_files/
OUT_DIR=$path/model_output_files/

job_number=$SLURM_ARRAY_TASK_ID
#source activate adiLab
python $path/MOT_model.py $job_number $JOB_DIR $OUT_DIR > /home-4/supadhy6\@jhu.edu/data/logs/velocity_based_corr/output_$job_number.log
