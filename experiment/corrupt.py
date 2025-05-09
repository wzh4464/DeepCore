from experiment_utils import initialize_flip_exp
from liveval.utils import count_flipped_in_lowest_scores
import pandas as pd
import torch

def run(args, checkpoint, start_exp, start_epoch):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Running corrupt experiment with {args.num_corrupt} corruptions on {args.num_scores} samples")
    scores = []
    for exp in range(start_exp, args.num_exp):
        (
            corrupted_train_dataset,
            test_loader,
            corrupted_indices,
            permuted_indices,
            corrupted_selection_from,
        ) = initialize_flip_exp(args, args.seed + exp)
        from liveval.methods.selection_methods import SELECTION_METHODS
        method_class = SELECTION_METHODS.get(args.selection)
        method = method_class(
            corrupted_train_dataset,
            args,
            args.fraction,
            args.seed,
            dst_test=test_loader.dataset,
            epochs=args.selection_epochs,
        )
        score = method.get_scores()
        try:
            df = pd.DataFrame(score.detach().numpy())
        except AttributeError:
            df = pd.DataFrame(score)
            score = torch.tensor(score)
        df.to_csv(f"{args.save_path}/corrupt_scores_{exp}.csv", index=False)
        scores.append(score)
        logger.debug(f"Scores: {scores}")
    average_score = torch.mean(torch.stack(scores), dim=0)
    df = pd.DataFrame(average_score.detach().numpy())
    df["index"] = corrupted_train_dataset.indices
    labels = []
    for idx in corrupted_train_dataset.indices:
        _, label, _ = corrupted_train_dataset.dataset[idx]
        labels.append(label)
    df["label"] = labels
    df.to_csv(f"{args.save_path}/average_corrupt_score_{args.timestamp}.csv", index=False)
    count_flipped_in_lowest_scores(logger, args, corrupted_indices, average_score) 