import wandb

run = wandb.init()
run.log({"test": 123})
run.finish()
