#!/bin/bash
#SBATCH --job-name=calibrate_filter
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=0-02:00:00
#SBATCH --output=log/calibrate_filter_%j.out
#SBATCH --error=log/calibrate_filter_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# What the tail-gap and unit-audit rules would remove, measured against the extracted events.
#
#   sbatch mimic4dataprep/tools/slurm_calibrate_outlier_filter.sh
#   SUBJECTS=0 sbatch mimic4dataprep/tools/slurm_calibrate_outlier_filter.sh   # every subject
#
# Submit from the TransEHR2 root, which is where the venv and the mimic4dataprep checkout live.
#
# NO PIPELINE STAGE IS RE-RUN. events.csv is step 1 output already on disk; the only work done
# to it is the mapping and cleaning that step 3 performs before a filter would sit. The raw
# MIMIC-IV tables are never opened, so this costs minutes rather than the hours step 1 does.
#
# A batch job rather than a login-node run because the accumulators hold a few hundred MiB
# before pandas is counted, against a 1 GB login node. Nothing is written outside the log.

set -uo pipefail

TRANSEHR2_ROOT="${TRANSEHR2_ROOT:-$(pwd)}"
M4DP_ROOT="${M4DP_ROOT:-${TRANSEHR2_ROOT}/mimic4dataprep}"
M4DP_DATA_DIR="${M4DP_DATA_DIR:-${HOME}/projects/p60290_2/mimic4dataprep/data/root}"
VARIABLE_MAP="${VARIABLE_MAP:-${M4DP_ROOT}/mimic4dataprep/resources/itemid_to_variable_map_used.csv}"
SUBJECTS="${SUBJECTS:-2000}"
GAP="${GAP:-0.5}"
MIN_FOLD="${MIN_FOLD:-3.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p log

for path in "${M4DP_DATA_DIR}" "${VARIABLE_MAP}" \
            "${M4DP_ROOT}/tools/calibrate_outlier_filter.py"; do
    if [ ! -e "${path}" ]; then
        echo "ERROR: missing ${path}" >&2
        exit 1
    fi
done

if [ ! -f venv/mimic4dataprep/bin/activate ]; then
    echo "ERROR: no virtual environment at venv/mimic4dataprep/bin/activate" >&2
    exit 1
fi
source venv/mimic4dataprep/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python -c 'import mimic4dataprep, numpy, pandas; print("mimic4dataprep importable")' || exit 1

echo "subjects root: ${M4DP_DATA_DIR}"
echo "variable map:  ${VARIABLE_MAP}"
echo "subjects:      ${SUBJECTS} (0 = all)"

cd "${M4DP_ROOT}" || exit 1

# shellcheck disable=SC2086  # EXTRA_ARGS is deliberately word-split
python tools/calibrate_outlier_filter.py \
    "${M4DP_DATA_DIR}" \
    --variable_map_file "${VARIABLE_MAP}" \
    --subjects "${SUBJECTS}" \
    --gap "${GAP}" \
    --min_fold "${MIN_FOLD}" ${EXTRA_ARGS}
STATUS=$?

echo ""
echo "======================================================================"
echo "Finished at $(date) with status ${STATUS}"
printf "Runtime: %02d:%02d:%02d\n" $((SECONDS/3600)) $((SECONDS%3600/60)) $((SECONDS%60))
echo "======================================================================"
echo ""
echo "Reading the output:"
echo "  * TAIL GAP: a variable removing more than the warn fraction is cutting real values,"
echo "    not errors. Raise --gap or --min_fold and look at it before adopting the rule."
echo "  * A 'none' cut means no detachment was found, which is the correct answer for a"
echo "    variable whose tail is continuous."
echo "  * UNIT AUDIT: the flagged itemid is the minority mode, not necessarily the wrong one."
echo "    Check both sides against UNITNAME in the variable map."
echo "  * Nothing was applied. This says what the rules would do."
exit ${STATUS}
