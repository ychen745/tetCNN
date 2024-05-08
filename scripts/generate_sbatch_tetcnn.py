import os

n_epoch = 200
lr = 0.001
wd = 0.0001

exp_name = 'tetcnn'
pos_folder = 'pos'
neg_folder = 'neg'

for cv in range(1):
    with open(os.path.join('sbatch_jobs', 'job_' + exp_name + '_' + str(cv) + '.sh'), 'w') as fout:
        out_lines = list()

        out_lines.append('#!/bin/bash\n')
        out_lines.append('#SBATCH -N 1                        # number of compute nodes')
        out_lines.append('#SBATCH -n 4                        # number of cores')
        out_lines.append('#SBATCH -p general                      # Use gpu partition')
        out_lines.append('#SBATCH --mem=128G')
        out_lines.append('#SBATCH -q public                 # Run job under wildfire QOS queue')
        out_lines.append('#SBATCH -G a100:1')
        out_lines.append('#SBATCH -t 1-00:00:00                  # wall time (D-HH:MM)')

        out_lines.append('#SBATCH -o /home/ychen855/tetCNN/scripts/job_logs/' + str(pos_name) + '_' + str(neg_name) + '_' + exp_name + '_' + str(cv) + '.out')
        out_lines.append('#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails')
        out_lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

        out_lines.append('\n')
        out_lines.append('module purge    # Always purge modules to ensure a consistent environment')
        out_lines.append('module load cuda-11.7.0-gcc-12.1.0')
        out_lines.append('source ~/.bashrc')
        out_lines.append('conda activate tetcnn')

        out_lines.append('python /home/ychen855/tetCNN/src/tetCNN.py --pos ' + pos_folder + ' --neg ' + neg_folder + ' --n_epoch ' + str(n_epoch) + ' --lr ' + str(lr) + ' --wd ' + str(wd) + ' --cv ' + str(cv) + ' --load')

        fout.write('\n'.join(out_lines) + '\n')
