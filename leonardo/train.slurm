#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=240
#SBATCH --gres=gpu:1
#SBATCH --job-name=ARM-TRAIN
#SBATCH --output=logs/%x-%j.out
#SBATCH --requeue
# Slurm submission script for training the ARM model on Leonardo.
#
# Note:
#  This script assumes you already bootstrapped the environment with:
#    $ bash leonardo/environment/setup.sh

ROOT=$(git rev-parse --show-toplevel)
cd "${ROOT}" || exit 1

# Load environment (modules, spack env, python venv)
source leonardo/environment/activate

export JAX_PLATFORMS=cuda,cpu
export JAX_TRACEBACK_FILTERING=off
export JAX_TRACEBACK_IN_LOCATIONS_LIMIT=-1
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"

python -m causal_conv_arm.train "${ROOT}/configs/train.toml" "${ROOT}/data" --jit --no-progress --no-download --log-level=info
