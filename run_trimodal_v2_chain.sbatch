#!/bin/bash
#SBATCH --job-name=trimodal-v2
#SBATCH --account=sfglab
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/mnt/evafs/groups/sfglab/ksengupta/pchic/pic_hic_ml/logs/trimodal_v2_chain/slurm_%j.log
#
# Self-resubmitting chain for notebooks/pangenomic_gat_pipeline_trimodal_1mb_SERVER_v2.py.
# Separate from run_trimodal_server_chain.sbatch: its own logs, crash counter and stop
# file, and the script writes to results/results_trimodal_1mb_v2_full/.
#
# - Each job runs for at most 5 days (the hopper partition limit). 10 min before the
#   limit SLURM sends USR1; the job resubmits itself (afterany dependency), stops
#   python, and exits.
# - --account=sfglab is required: the default "null" account has MaxSubmit=0.
# - The pipeline checkpoints atomically (tmp + fsync + rename) after every epoch and
#   stores finished folds, so the next job re-runs the script and continues.
# - A crash (non-zero exit, e.g. OOM) is retried up to MAX_CRASH_RETRIES times in a row;
#   the counter resets after any job that reaches the time limit.
# - Chain stops when python exits 0 (writes logs/trimodal_v2_chain/DONE), when the crash
#   retry budget is used up, or when the file STOP_TRIMODAL_V2_CHAIN exists in the
#   project root.
#
# Start:   sbatch run_trimodal_v2_chain.sbatch
# Stop:    touch STOP_TRIMODAL_V2_CHAIN   (then scancel the running/pending job if wanted)

PROJ=/mnt/evafs/groups/sfglab/ksengupta/pchic/pic_hic_ml
SCRIPT=pangenomic_gat_pipeline_trimodal_1mb_SERVER_v2.py
LOGDIR=$PROJ/logs/trimodal_v2_chain
LOG=$LOGDIR/pipeline_job${SLURM_JOB_ID}.log
CRASH_COUNT_FILE=$LOGDIR/crash_count
MAX_CRASH_RETRIES=3
mkdir -p "$LOGDIR"

source /mnt/evafs/groups/sfglab/ksengupta/soft/miniforge3/etc/profile.d/conda.sh
conda activate pchic

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PANGENOMIC_ROOT=$PROJ

resubmit() {
    if [ -e "$PROJ/STOP_TRIMODAL_V2_CHAIN" ]; then
        echo "[$(date)] STOP_TRIMODAL_V2_CHAIN present -- not resubmitting"; return
    fi
    NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$PROJ/run_trimodal_v2_chain.sbatch")
    echo "[$(date)] resubmitted as job $NEXT"
    echo "$NEXT" > "$LOGDIR/latest_job_id"
}

on_timeout() {
    echo "[$(date)] got USR1 (time limit approaching) -- restarting from checkpoint"
    echo 0 > "$CRASH_COUNT_FILE"
    resubmit
    kill -TERM "$PID" 2>/dev/null
    wait "$PID"
    echo "[$(date)] pipeline stopped; next job resumes from latest checkpoint"
    exit 0
}
trap on_timeout USR1

echo "[$(date)] job $SLURM_JOB_ID on $(hostname -s), GPU $CUDA_VISIBLE_DEVICES"
echo "[$(date)] pipeline log -> $LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "existing checkpoints:"; ls -la "$PROJ/results/results_trimodal_1mb_v2_full/checkpoints/" 2>/dev/null

cd "$PROJ/notebooks"
python -u "$SCRIPT" > "$LOG" 2>&1 &
PID=$!
wait "$PID"
RC=$?
echo "[$(date)] pipeline exited with code $RC"

if [ $RC -eq 0 ]; then
    echo "[$(date)] pipeline FINISHED -- chain complete"
    touch "$LOGDIR/DONE"
    rm -f "$CRASH_COUNT_FILE"
    exit 0
fi

CRASHES=$(( $(cat "$CRASH_COUNT_FILE" 2>/dev/null || echo 0) + 1 ))
echo $CRASHES > "$CRASH_COUNT_FILE"
if [ $CRASHES -le $MAX_CRASH_RETRIES ]; then
    echo "[$(date)] crash $CRASHES/$MAX_CRASH_RETRIES -- restarting from checkpoint"
    resubmit
else
    echo "[$(date)] $CRASHES consecutive crashes -- giving up; check $LOG"
fi
exit $RC
