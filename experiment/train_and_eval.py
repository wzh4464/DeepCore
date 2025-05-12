from experiment.experiment_utils import (
    initialize_dataset_and_model,
    train_and_evaluate_model,
    print_experiment_info,
    setup_checkpoint_name,
)
import time

def run(args, checkpoint, start_exp, start_epoch):
    for exp in range(start_exp, args.num_exp):
        checkpoint_name = setup_checkpoint_name(args, exp) if args.save_path != "" else ""
        print_experiment_info(args, exp, checkpoint_name)
        train_loader, test_loader, if_weighted, subset, selection_args = initialize_dataset_and_model(args, checkpoint)
        models = [args.model] + ([model for model in args.cross if model != args.model] if isinstance(args.cross, list) else [])
        for model in models:
            if len(models) > 1:
                import logging
                logging.getLogger(__name__).info(f"| Training on model {model}")
            train_and_evaluate_model(
                args, exp, start_epoch, train_loader, test_loader, subset, selection_args, checkpoint_name, model, checkpoint
            )
        start_epoch = 0
        checkpoint = {}
        time.sleep(2) 