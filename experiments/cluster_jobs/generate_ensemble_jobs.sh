#!/bin/bash
# ============================================================
# generate_ensemble_jobs.sh
# ============================================================
# Generates LSF scripts for ensemble training, evaluation,
# and prediction.
#
# Usage:
#   bash generate_ensemble_jobs.sh
#   bsub < ensemble_train.lsf
#   # After training completes:
#   bsub < ensemble_evaluate.lsf
#   # For new predictions:
#   bsub < ensemble_predict.lsf

# ---- EDIT THESE PATHS ----
DATA_CSV="${SPECTRA_ROOT}/app/ImmunoStruct/data/training_rosetta.csv"
PROJECT_DIR="${SPECTRA_ROOT}/project/TCRpMHC"
OUT_DIR="${PROJECT_DIR}/outputs/ensemble"
ESM_CKPT="${SPECTRA_ROOT}/data/models/facebook/esm2_t12_35M_UR50D"
CONDA_ENV="immunostruct"
# For prediction on new data (edit when ready):
NEW_DATA_CSV="${DATA_CSV}"
# --------------------------

# ============================================================
# Job 1: Train 5 ensemble members
# ============================================================
cat > ensemble_train.lsf << ENDOFLSF
#!/bin/bash
#BSUB -J ens_train
#BSUB -q egpu
#BSUB -n 8
#BSUB -M 64
#BSUB -R "rusage[mem=64]"
#BSUB -gpu "num=1:gmem=24"
#BSUB -W 72:00
#BSUB -o ${OUT_DIR}/train_job_%J.out
#BSUB -e ${OUT_DIR}/train_job_%J.err
#BSUB -cwd ${PROJECT_DIR}

source activate ${CONDA_ENV}
mkdir -p ${OUT_DIR}

echo "======================================"
echo "  Ensemble Training: 5 Members"
echo "  Start: \$(date)"
echo "  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================"

cd "\$HOME/app/ImmunoStruct/esm_rosetta"
python ensemble_tcr.py train \\
    --data_csv ${DATA_CSV} \\
    --esm_checkpoint ${ESM_CKPT} \\
    --n_ensemble 5 \\
    --seeds 42 123 456 789 2024 \\
    --peptide_split \\
    --d_model 128 \\
    --n_cross_heads 4 \\
    --n_cross_layers 2 \\
    --d_rosetta 64 \\
    --d_fused 256 \\
    --dropout 0.2 \\
    --batch_size 16 \\
    --epochs 30 \\
    --lr 3e-4 \\
    --esm_lr 1e-5 \\
    --patience 7 \\
    --crystal_weight 5.0 \\
    --unfreeze_b 10 \\
    --unfreeze_c 20 \\
    --num_workers 4 \\
    --ckpt_dir ${OUT_DIR} \\
    --progress_bar

echo "======================================"
echo "  Training complete: \$(date)"
echo "======================================"
ENDOFLSF

echo "Generated: ensemble_train.lsf"

# ============================================================
# Job 2: Evaluate ensemble on test set
# ============================================================
cat > ensemble_evaluate.lsf << ENDOFLSF
#!/bin/bash
#BSUB -J ens_eval
#BSUB -q egpu
#BSUB -n 8
#BSUB -M 64
#BSUB -R "rusage[mem=64]"
#BSUB -gpu "num=1:gmem=24"
#BSUB -W 2:00
#BSUB -o ${OUT_DIR}/eval_job_%J.out
#BSUB -e ${OUT_DIR}/eval_job_%J.err
#BSUB -cwd ${PROJECT_DIR}

source activate ${CONDA_ENV}

echo "======================================"
echo "  Ensemble Evaluation"
echo "  Start: \$(date)"
echo "======================================"

cd "\$HOME/app/ImmunoStruct/esm_rosetta"
python ensemble_tcr.py evaluate \\
    --data_csv ${DATA_CSV} \\
    --esm_checkpoint ${ESM_CKPT} \\
    --peptide_split \\
    --d_model 128 \\
    --n_cross_heads 4 \\
    --n_cross_layers 2 \\
    --d_rosetta 64 \\
    --d_fused 256 \\
    --batch_size 16 \\
    --num_workers 4 \\
    --ckpt_dir ${OUT_DIR}

echo "======================================"
echo "  Evaluation complete: \$(date)"
echo "  Results: ${OUT_DIR}/ensemble_eval_results.json"
echo "======================================"
ENDOFLSF

echo "Generated: ensemble_evaluate.lsf"

# ============================================================
# Job 3: Predict on new data
# ============================================================
cat > ensemble_predict.lsf << ENDOFLSF
#!/bin/bash
#BSUB -J ens_pred
#BSUB -q egpu
#BSUB -n 8
#BSUB -M 64
#BSUB -R "rusage[mem=64]"
#BSUB -gpu "num=1:gmem=24"
#BSUB -W 2:00
#BSUB -o ${OUT_DIR}/pred_job_%J.out
#BSUB -e ${OUT_DIR}/pred_job_%J.err
#BSUB -cwd ${PROJECT_DIR}

source activate ${CONDA_ENV}

echo "======================================"
echo "  Ensemble Prediction"
echo "  Start: \$(date)"
echo "======================================"

cd "\$HOME/app/ImmunoStruct/esm_rosetta"
python ensemble_tcr.py predict \\
    --input_csv ${NEW_DATA_CSV} \\
    --esm_checkpoint ${ESM_CKPT} \\
    --d_model 128 \\
    --n_cross_heads 4 \\
    --n_cross_layers 2 \\
    --d_rosetta 64 \\
    --d_fused 256 \\
    --batch_size 16 \\
    --num_workers 4 \\
    --ckpt_dir ${OUT_DIR} \\
    --output_csv ${OUT_DIR}/predictions.csv \\
    --threshold 0.5

echo "======================================"
echo "  Prediction complete: \$(date)"
echo "  Output: ${OUT_DIR}/predictions.csv"
echo "======================================"
ENDOFLSF

echo "Generated: ensemble_predict.lsf"

echo ""
echo "========================================="
echo "  WORKFLOW"
echo "========================================="
echo ""
echo "  Step 1: Train ensemble"
echo "    bsub < ensemble_train.lsf"
echo ""
echo "  Step 2: Evaluate (after training completes)"
echo "    bsub < ensemble_evaluate.lsf"
echo ""
echo "  Step 3: Predict on new data"
echo "    # Edit NEW_DATA_CSV in this script first"
echo "    bsub < ensemble_predict.lsf"
echo ""
echo "  Results will be in: ${OUT_DIR}/"
echo "    ensemble_config.json       - member checkpoints and config"
echo "    ensemble_eval_results.json - metrics for all strategies"
echo "    predictions.csv            - per-sample predictions"
echo "========================================="