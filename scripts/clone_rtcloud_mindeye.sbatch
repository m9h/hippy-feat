#!/bin/bash
#SBATCH --job-name=clone-rtcloud-mindeye
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/data/derivatives/rtmindeye_paper/logs/%x-%j.out

# Clone the canonical rtcloud-projects/mindeye repo so we have the real
# model.py (MindEye2 architecture) + any inference scripts not in the
# cobalt-cosmos playground copy at ~/.gemini/.../mindeye/.
# Also clone Stability-AI/generative-models (sgm/) needed for SDXL unCLIP
# if we end up doing reconstruction; retrieval-only path can skip it.

set -euo pipefail

REPOS=/data/derivatives/rtmindeye_paper/repos
mkdir -p "$REPOS" /data/derivatives/rtmindeye_paper/logs

cd "$REPOS"
if [ -d rtcloud-projects/.git ]; then
    git -C rtcloud-projects fetch --depth 1 origin main
    git -C rtcloud-projects reset --hard origin/main
else
    git clone --depth 1 https://github.com/brainiak/rtcloud-projects.git
fi

echo "rtcloud-projects/mindeye tree:"
ls rtcloud-projects/mindeye/ 2>/dev/null
echo "---scripts:"
ls rtcloud-projects/mindeye/scripts/ 2>/dev/null | head -20
echo "---models.py present?"
find rtcloud-projects -name "models.py" | head
