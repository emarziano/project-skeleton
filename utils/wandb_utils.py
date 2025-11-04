import wandb

def setup_wandb(project: str, config: dict):
    wandb.init(project=project, config=config)
    wandb.watch_called = False
    wandb.watch(models=None, log="all")
