run:

setup.sh

bpe_and_fairseq_preprocess_10splits.py 

slurm_training.slurm (if have cluster, otherwise just run training_split_data.py)

generate_predictions.slurm (if have cluster, otherwise just run generate_predictions.py)

algorithm1.slurm  (if have cluster, otherwise just run algorithm1_1.py, algorithm1_2.py, algorithm1_3.py, algorithm1_4.py)

merge_algorithm1.py

algorithm2.py (will fail, this is expected)

generate_predictions_perturbed.slurm (if have cluster, otherwise just run generate_perturbation_preds.py)

algorithm2.py (should not fail this time)


