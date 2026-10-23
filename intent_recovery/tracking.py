"""Optional W&B tracking; only numeric metrics and configuration are logged."""


def add_tracking_args(parser):
    parser.add_argument("--wandb", action="store_true", help="enable W&B tracking")
    parser.add_argument("--wandb-project", default="intent-recovery")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)


def init_tracking(args, config, job_type):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                      name=args.wandb_name, mode=args.wandb_mode,
                      config=config, job_type=job_type)


def metric_log(metrics, prefix):
    values = {}
    for key, value in metrics.items():
        if isinstance(value, (list, tuple)):
            values.update({f"{prefix}/{key}/r{i+1}": v for i, v in enumerate(value)})
        else:
            values[f"{prefix}/{key}"] = value
    return values
