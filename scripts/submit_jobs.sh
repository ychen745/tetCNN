for filename in sbatch_jobs/*.sh; do
    # echo $filename
    sbatch $filename
done