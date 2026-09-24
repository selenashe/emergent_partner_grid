#!/bin/bash

echo "Pulling latest changes from Git..."
git -C /scratch/gpfs/$USER/overcooked_jax_meta pull

SEEDS=$(seq 10)

for seed in ${SEEDS[@]}; do
  TIMESTAMP=$(date +%Y%m%d_%H%M%S_%N)
  SNAPSHOT_DIR="/scratch/gpfs/$USER/overcooked_jax_meta_snapshot_$TIMESTAMP"
  rsync -a --delete /scratch/gpfs/$USER/overcooked_jax_meta/ "$SNAPSHOT_DIR/"

  sed -i -E "s/(\"SEED\":\s*)[0-9]+/\1$seed/" "$SNAPSHOT_DIR/baselines/IPPO/config/ippo_rnn_overcooked_v2.yaml"

  sbatch --parsable \
    --export=SNAPSHOT_DIR="$SNAPSHOT_DIR",SEED="$seed" \
    run_apptainer_mounted.sh
done