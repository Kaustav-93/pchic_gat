#!/bin/bash
#SBATCH --job-name=trimodal-v3
#SBATCH --account=sfglab
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --output=/mnt/evafs/groups/sfglab/ksengupta/pchic/pic_hic_ml/logs/trimodal_v3_chain/slurm_%j.log
#
# Self-resubmitting job for notebooks/pangenomic_gat_pipeline_trimodal_1mb_SERVER_v3.py.
# One main job (ROLE=main) plus any number of Stage-A workers (ROLE=worker), which
# share the 15 Stage-A splits through lock files in checkpoints/stage_a_pool/.
#
# Start:   ROLE=main sbatch run_trimodal_v3_chain.sbatch
#          for i in $(seq 1 14); do ROLE=worker sbatch run_trimodal_v3_chain.sbatch; done
# Do NOT use sbatch --export here: it makes SLURM fetch the login environment, which
# fails on the dgx nodes (job held as user_env_retrieval_failed_requeued_held).
# Stop:    touch STOP_TRIMODAL_V3_CHAIN   (then scancel the jobs)

PROJ=/mnt/evafs/groups/sfglab/ksengupta/pchic/pic_hic_ml
SCRIPT=pangenomic_gat_pipeline_trimodal_1mb_SERVER_v3.py
ROLE=${ROLE:-main}
LOGDIR=$PROJ/logs/trimodal_v3_chain
LOG=$LOGDIR/pipeline_${ROLE}_job${SLURM_JOB_ID}.log
CRASH_COUNT_FILE=$LOGDIR/crash_count_${ROLE}_${CHAIN_ID:-$SLURM_JOB_ID}
export CHAIN_ID=${CHAIN_ID:-$SLURM_JOB_ID}
MAX_CRASH_RETRIES=3
mkdir -p "$LOGDIR"

source /mnt/evafs/groups/sfglab/ksengupta/soft/miniforge3/etc/profile.d/conda.sh
conda activate pchic

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export PANGENOMIC_ROOT=$PROJ
[ "$ROLE" = worker ] && export PANGENOMIC_STAGEA_ONLY=1

resubmit() {
    if [ -e "$PROJ/STOP_TRIMODAL_V3_CHAIN" ]; then
        echo "[$(date)] STOP_TRIMODAL_V3_CHAIN present -- not resubmitting"; return
    fi
    NEXT=$(ROLE=$ROLE CHAIN_ID=$CHAIN_ID sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$PROJ/run_trimodal_v3_chain.sbatch")
    echo "[$(date)] resubmitted as job $NEXT"
}

on_timeout() {
    echo "[$(date)] got USR1 (time limit approaching) -- restarting from checkpoint"
    echo 0 > "$CRASH_COUNT_FILE"
    resubmit
    kill -TERM "$PID" 2>/dev/null
    wait "$PID"
    exit 0
}
trap on_timeout USR1

echo "[$(date)] $ROLE job $SLURM_JOB_ID on $(hostname -s), GPU $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$PROJ/notebooks"
python -u "$SCRIPT" > "$LOG" 2>&1 &
PID=$!
wait "$PID"
RC=$?
echo "[$(date)] pipeline exited with code $RC"

if [ $RC -eq 0 ]; then
    [ "$ROLE" = main ] && touch "$LOGDIR/DONE"
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
