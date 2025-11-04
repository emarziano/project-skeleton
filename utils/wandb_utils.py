import wandb

def setup_wandb(project: str, config: dict):
    wandb.init(project=project, config=config)
