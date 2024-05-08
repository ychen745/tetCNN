import os
import shutil

data_folder = '/data/hohokam/Yanxi/Data/NACC_hippo'
matlab_folder = '/home/ychen855/tetCNN/src/matlab'

lines = list()
lines.append('#!/bin/bash\n')
lines.append('#SBATCH -n 4')
lines.append('#SBATCH -t 0-04:00:00')
lines.append('#SBATCH -o /home/ychen855/tetCNN/scripts/job_logs/slurm.%j.out')
lines.append('#SBATCH -e /home/ychen855/tetCNN/scripts/job_logs/slurm.%j.err')
lines.append('#SBATCH --mail-type=NONE')
lines.append('#SBATCH --mail-user=ychen855@asu.edu\n')

lines.append('module purge    # Always purge modules to ensure a consistent environment')
lines.append('module load matlab/2022a\n')

for subfolder in os.listdir(data_folder):
	fout = open(os.path.join(tetgen_folder, 'sbatch_jobs', subfolder + '_' + half + '.sh'), 'w')
	out_lines = []
	start_folder = os.path.join(data_folder, subfolder)
	start_folder = start_folder if start_folder[-1] != '/' else start_folder[:-1]
	out_lines.append('cd ' + os.path.join(tetgen_folder, 'matlab') + '\n')
	out_lines.append('matlab -batch \"calc_LBO_lump_' + half + ' \"')
	fout.write('\n'.join(lines + out_lines))
