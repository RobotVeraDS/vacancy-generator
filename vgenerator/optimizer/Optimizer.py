import torch

class Optimizer(object):
    def __init__(self, optim, scheduler):
        self.optim = optim
        self.scheduler = scheduler


    def step(self):
        self.optim.step()


    def update(self, loss):
        if self.scheduler is None:
            return

        if isinstance(self.scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(loss)
        else:
            self.scheduler.step()
