from basictorch_v2.trainer import Trainer

class TPTrainer(Trainer):
    def __init__(self, name, args, outM=None, model=None, default_args={}, args_names=[], **kwargs):
        super().__init__(name, args, outM, model, default_args, args_names, **kwargs)
    
    def get_losses(self, batch_data, *args, **kwargs):
        return self.model(batch_data, *args, **kwargs)
