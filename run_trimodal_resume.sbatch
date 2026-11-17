#!/bin/bash
#SBATCH --job-name=trimodal-resume
#SBATCH --account=sfglab
#SBATCH --partition=hopper-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=360G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/mnt/evafs/groups/sfglab/ksengupta/pchic/pic_hic_ml/logs/trimodal_resume_slurm_%j.log
#
# Runs pangenomic_gat_pipeline_trimodal_1mb_SERVER (resume script) and, when the
# 24h wallclock is about to run out, resubmits itself. The pipeline checkpoints
# every epoch, so each new job picks up from the last finished epoch.
# Chain stops when the pipeline exits 0 (finished) or crashes (non-timeout error),
# or when the file STOP_TRIMODAL_CHAIN exists in the project root.

PROJ=/mnt/evafs/groups/sfglab/ksengupta/pchic/pic_hic_ml
LOG=$PROJ/logs/trimodal_1mb_SERVER_resume_job${SLURM_JOB_ID}.log

source /mnt/evafs/groups/sfglab/ksengupta/soft/miniforge3/etc/profile.d/conda.sh
conda activate pchic
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PIPELINE_NUM_WORKERS=8   # 32 workers OOM-killed jobs 1917691 + 1920537 (copy-on-write growth)

resubmit() {
    if [ -e "$PROJ/STOP_TRIMODAL_CHAIN" ]; then
        echo "[$(date)] STOP_TRIMODAL_CHAIN present -- not resubmitting"; return
    fi
    NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$PROJ/run_trimodal_resume.sbatch")
    echo "[$(date)] wallclock nearly up -- resubmitted as job $NEXT"
}
on_timeout() {
    echo "[$(date)] got USR1 (15 min left)"
    resubmit
    kill -TERM "$PID" 2>/dev/null
    wait "$PID"
    exit 0
}
trap on_timeout USR1

echo "[$(date)] job $SLURM_JOB_ID on $(hostname -s), GPU $CUDA_VISIBLE_DEVICES, log -> $LOG"
cd "$PROJ/notebooks"
python -u pangenomic_gat_pipeline_trimodal_1mb_SERVER_resume.py > "$LOG" 2>&1 &
PID=$!
wait "$PID"
RC=$?
# `wait` returns >128 when interrupted by the trapped signal; on_timeout handles that path.
echo "[$(date)] pipeline exited with code $RC"
if [ $RC -eq 0 ]; then
    echo "[$(date)] pipeline FINISHED -- chain complete"
    touch "$PROJ/logs/trimodal_1mb_SERVER_resume.DONE"
fi
