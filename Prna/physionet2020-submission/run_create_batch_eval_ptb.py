#!/usr/bin/env python3

import argparse

BATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=singularity_eval_ptb_split{split_id}_of_{num_splits}
#SBATCH --output=singularity_eval_ptb_split{split_id}_of_{num_splits}.out
#SBATCH --error=singularity_eval_ptb_split{split_id}_of_{num_splits}.err
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=all
#SBATCH --account=nlp
#SBATCH --partition=jag-important
#SBATCH --ntasks=1
#SBATCH --mem=40g
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:titanrtx:1  # Only if GPU is needed

# Optional: Load singularity module
# module load singularity

# Paths
CONTAINER_DIR="my_container_dir"
BIND_PATHS="/juice2/scr2/kelvinkn/other_work/edwards/ptb/converted_ptxbl_records500:/physionet/inputs,/juice2/scr2/kelvinkn/other_work/edwards/ptb/prna_ptb500_outputs:/physionet/outputs,/juice2/scr2/kelvinkn/other_work/edwards/Prna/trained_model:/physionet/trained_model"

echo "Starting run ptb split {split_id} of {num_splits}" 
# Run the command inside the container
# IMPORTANT ADD -u TO PYTHON COMMAND TO UNBUFFER OUTPUTS
singularity exec --nv \
  --writable-tmpfs \
  --bind "$BIND_PATHS" \
  "$CONTAINER_DIR" \
  bash -c "cd /physionet && python -u driver.py trained_model/ inputs/ outputs/ 40 {eval_fraction:.6f} {split_id}"

"""

def main():
    parser = argparse.ArgumentParser(description="Generate batch files for PTB driver.py split run.")
    parser.add_argument("--num_splits", type=int, required=True, help="Number of splits (ex: 3 for thirds, 4 for fourths, etc.)")

    args = parser.parse_args()

    for split_id in range(args.num_splits):
        eval_fraction = 1.0 / args.num_splits

        batch_script_text = BATCH_TEMPLATE.format(
            num_splits=args.num_splits,
            split_id=split_id,
            eval_fraction=eval_fraction
        )

        out_filename = f"batch_split{split_id + 1}_of_{args.num_splits}.sh"

        with open(out_filename, "w") as f:
            f.write(batch_script_text)
    
        print(f"Generated batch script: {out_filename}")

if __name__ == "__main__":
    main()
