. /usr/local/anaconda3/etc/profile.d/conda.sh 
conda activate clip

srun -Q --immediate=10 --partition=all_serial --gres=gpu:1 --time 10:00 --pty bash

