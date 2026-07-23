#!/bin/bash
# Generate one LSF script per ablation mode (A-H) with per-mode resources,
# then submit them as independent single-GPU jobs.
#   bash hpc/generate_ablation_jobs.sh && for f in ablation_*.lsf; do bsub < "$f"; done
set -euo pipefail
: "${SPECTRA_DATA_DIR:?set SPECTRA_DATA_DIR}"
OUT_DIR="${SPECTRA_OUT_DIR:-$PWD/outputs/ablation}"
MODES=(A B C D E F G H)
WALL=(6:00 12:00 12:00 16:00 16:00 6:00 8:00 8:00)   # 4-chain modes cost more
MEM=(48 64 64 64 64 48 48 48)                         # 4 ESM passes need more RAM
for i in "${!MODES[@]}"; do
  m=${MODES[$i]}
  cat > "ablation_${m}.lsf" <<LSF
#!/bin/bash
#BSUB -W ${WALL[$i]}
#BSUB -q egpu
#BSUB -gpu num=1:gmem=32
#BSUB -n 1
#BSUB -M ${MEM[$i]}
#BSUB -R rusage[mem=${MEM[$i]}]
#BSUB -J ablation_${m}
#BSUB -o logs/%J_${m}.out
#BSUB -e logs/%J_${m}.err
module purge; module load miniforge3 cuda12.2/toolkit/12.2.2 gcc/12.4.0
eval "\$(conda shell.bash hook)"; conda activate spectra
python -m spectra.training.ablation --modes ${m} \\
    --data_csv "\$SPECTRA_DATA_DIR/training.csv" \\
    --devices 1 --out_dir "${OUT_DIR}"
LSF
done
echo "Wrote ablation_{A..H}.lsf"
