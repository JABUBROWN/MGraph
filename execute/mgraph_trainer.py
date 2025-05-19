import torch.nn.functional as F

from execute.base_trainer import BaseTrainer


class MGraphTrainer(BaseTrainer):

    def __init__(self, args, model, optimizer, lr_scheduler, loss, train_dataloader, val_dataloader, test_dataloader, scaler, logger):
        super(MGraphTrainer, self).__init__(args, model, optimizer, lr_scheduler, loss, train_dataloader, val_dataloader, test_dataloader, scaler, logger)

    def before_model_forward(self, X, y, stage):
        X = X[..., [0, 4, 5]]  # idx 1 and 2 are s-t marker
        data = X[..., 0:1]
        tod = F.one_hot(X[..., 0, 1:2].long(), num_classes=288).squeeze(2).float().to(self.args.device)
        dow = F.one_hot(X[..., 0, 2:3].long(), num_classes=7).squeeze(2).float().to(self.args.device)
        X = [data, tod, dow]
        return X

    def after_model_forward(self, out, y, stage):
        out = self.scaler.reverse(out)
        y = self.scaler.reverse(y)
        return out, y

    def train(self):
        return super().base_train(self.before_model_forward, self.after_model_forward)
