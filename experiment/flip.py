from experiment_utils import initialize_flip_exp
from liveval.utils import count_flipped_in_lowest_scores
from experiment_utils import find_found_flipped_indices
import pandas as pd
import torch

def run(args, checkpoint, start_exp, start_epoch):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Running flip experiment with {args.num_flip} flips on {args.num_scores} samples")
    scores = []
    for exp in range(start_exp, args.num_exp):
        (
            flipped_train_dataset,
            test_loader,
            flipped_indices,
            permuted_indices,
            flipped_selection_from,
        ) = initialize_flip_exp(args, args.seed + exp)
        method_class = args.selection if isinstance(args.selection, type) else None
        from liveval.methods.selection_methods import SELECTION_METHODS
        method_class = SELECTION_METHODS.get(args.selection)
        method = method_class(
            flipped_train_dataset,
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
        df.to_csv(f"{args.save_path}/flip_scores_{exp}.csv", index=False)
        scores.append(score)
        logger.debug(f"Scores: {scores}")

        # 统一统计空间，确保 flipped_indices 都在 flipped_train_dataset.indices 里
        assert all(idx in flipped_train_dataset.indices for idx in flipped_indices), "flipped_indices 必须全部在 flipped_train_dataset.indices 中"
        # 统计被找到的反转点（用新工具函数）
        found_flipped_indices = find_found_flipped_indices(score, flipped_indices)
        pd.DataFrame({
            "found_flipped_indices": found_flipped_indices
        }).to_csv(f"{args.save_path}/found_flipped_indices_{args.timestamp}_{exp}.csv", index=False)

        # 构建去掉这些点的新训练集
        from torch.utils.data import Subset
        remaining_indices = [i for i in flipped_train_dataset.indices if i not in found_flipped_indices]
        new_train_dataset = Subset(flipped_train_dataset.dataset, remaining_indices)

        # 用新数据集重新训练，记录loss和accuracy
        retrain_method = method_class(
            new_train_dataset,
            args,
            args.fraction,
            args.seed,
            dst_test=test_loader.dataset,
            epochs=args.selection_epochs,
        )
        step_losses = []
        epoch_accuracies = []

        # hook loss/accuracy
        def after_loss_hook(outputs, loss, targets, batch_inds, epoch):
            step_losses.append(loss.item())
        def after_epoch_hook():
            if hasattr(retrain_method, 'test'):
                acc = getattr(retrain_method, 'last_test_acc', None)
                if acc is not None:
                    epoch_accuracies.append(acc)
        retrain_method.after_loss = after_loss_hook
        retrain_method.after_epoch = after_epoch_hook

        # 训练

        test_acc = retrain_method.train(args.selection_epochs, remaining_indices, test_loader)
        logger.info(f"Test accuracy: {test_acc:.4f}")

        # 保存loss/accuracy
        pd.DataFrame({"step_loss": step_losses}).to_csv(f"{args.save_path}/step_losses_{args.timestamp}_{exp}.csv", index=False)
        pd.DataFrame({"epoch_accuracy": epoch_accuracies}).to_csv(f"{args.save_path}/epoch_accuracies_{args.timestamp}_{exp}.csv", index=False)
    average_score = torch.mean(torch.stack(scores), dim=0)
    df = pd.DataFrame(average_score.detach().numpy())
    df["index"] = flipped_train_dataset.indices
    labels = []
    for idx in flipped_train_dataset.indices:
        _, label, _ = flipped_train_dataset.dataset[idx]
        labels.append(label)
    df["label"] = labels
    df.to_csv(f"{args.save_path}/average_score_{args.timestamp}.csv", index=False)
    count_flipped_in_lowest_scores(logger, args, flipped_indices, average_score) 
