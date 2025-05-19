import copy
import os.path
import statistics
import time

import torch
import numpy as np
from tqdm import tqdm

from util.metrics import cal_metrics


TRAIN = 0
VAL = 1
TEST = 2

class BaseTrainer:

    def __init__(self, args, model, optimizer, lr_scheduler, loss, train_dataloader, val_dataloader, test_dataloader, scaler, logger):
        super(BaseTrainer, self).__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.scaler = scaler
        self.logger = logger

    def base_train(self, before_model_forward, after_model_forward):
        best_loss = float('inf')
        best_epoch = 0
        best_model = None
        patience = 0
        for epoch in range(self.args.epoch):

            # [ Training Stage ]
            self.model.train()
            batch_number = len(self.train_dataloader)
            epoch_train_loss = 0
            bar = tqdm(enumerate(self.train_dataloader))  # Batch Processing Bar
            desc = None
            for batch, (X, y) in bar:
                self.optimizer.zero_grad()

                X = before_model_forward(X, y, TRAIN)
                if isinstance(X, torch.Tensor):
                    print("Shape of X before model:", X.shape)
                out = self.model(X)  # X is usually (B, T, N, C)
                out, y = after_model_forward(out, y, TRAIN)

                loss = self.loss(out, y[..., 0:1])
                loss.backward()

                if self.args.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                self.optimizer.step()

                epoch_train_loss += loss.item()  # Accumulate batch loss onto epoch loss
                desc = "[Train] Epoch: {}, Batch: {}/{}, Average Loss: {:.4f}, lr: {:.6f}".format(
                    epoch + 1, batch + 1, batch_number, epoch_train_loss / (batch + 1), self.optimizer.param_groups[0]['lr']
                )
                bar.set_description(desc)
            bar.close()
            epoch_train_loss /= batch_number  # Calculate the average.
            self.logger.log(desc)

            if self.args.lr_scheduler_name == "ExponentialLR":
                if (epoch+1) % int(self.args.lr_scheduler.split(',')[-1]) == 0:
                    self.lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # [ Validation Stage ]
            self.model.eval()
            batch_number = len(self.val_dataloader)
            epoch_val_loss = 0
            with torch.no_grad():
                bar = tqdm(enumerate(self.val_dataloader))  # Batch Processing Bar
                for batch, (X, y) in bar:
                    X = before_model_forward(X, y, VAL)
                    out = self.model(X)
                    out, y = after_model_forward(out, y, VAL)

                    loss = self.loss(out, y[..., 0:1])
                    epoch_val_loss += loss.item()  # Accumulate batch loss onto epoch loss
                    desc = "[ Val ] Epoch: {}, Batch: {}/{}, Average Loss: {:.4f}".format(
                        epoch + 1, batch + 1, batch_number, epoch_val_loss / (batch + 1))
                    bar.set_description(desc)
                bar.close()
                epoch_val_loss /= batch_number  # Calculate the average.
                self.logger.log(desc)

            # [ Update Best ]
            time.sleep(1)
            if epoch_val_loss < best_loss:
                best_model = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                best_loss = epoch_val_loss
                patience = 0
                self.logger.log_and_print("[Result] Epoch: {}, Current best model!".format(epoch + 1))
            else:
                patience += 1
            if self.args.early_stop and patience >= self.args.early_stop_patience:
                self.logger.log_and_print(
                    "[Result] Epoch: {}, Reached the maximum patience step: {}".format(
                        epoch, self.args.early_stop_patience))
                break
            time.sleep(1)

        # [ Save Best Model as .pth ]
        checkpoint_path = os.path.join(self.args.root_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path, f"mgraph_{self.args.data_name}_seed{self.args.seed}.pth")
        torch.save(best_model, checkpoint_file)
        self.logger.log_and_print(f"[Result] Saved best model to {checkpoint_file}")

        # [ Test Stage ]
        self.logger.log_and_print("[ Test ] Best model epoch is {}, when the validation loss is {:.6f}".format(best_epoch, best_loss))
        self.test(best_model, before_model_forward, after_model_forward)

    def test(self, best_model, before_model_forward, after_model_forward):
        self.model.load_state_dict(best_model)
        self.model.eval()
        batch_number = len(self.test_dataloader)
        y_pred = []
        y_true = []
        save_time = []
        save_space = []
        save_label = []
        save_predict = []

        with torch.no_grad():
            bar = tqdm(enumerate(self.test_dataloader))
            for batch, (X, y) in bar:
                X = before_model_forward(X, y, TEST)
                out = self.model(X)
                out, y = after_model_forward(out, y, TEST)

                y_pred.append(out)
                y_true.append(y[..., 0:1])
                bar.set_description("[ Test ] Batch: {}/{}".format(batch + 1, batch_number))

                save_time.append(y[..., 1])
                save_space.append(y[..., 2])
                save_label.append(y[..., 0])
                save_predict.append(out[..., 0])
            bar.close()

        # output npz (time, space, label, predict)
        save_time = np.array(np.round(torch.cat(save_time, dim=0).cpu().numpy()), dtype='int32')
        save_space = np.array(np.round(torch.cat(save_space, dim=0).cpu().numpy()), dtype='int32')
        save_label = np.array(torch.cat(save_label, dim=0).cpu().numpy(), dtype='float32')
        save_predict = np.array(torch.cat(save_predict, dim=0).cpu().numpy(), dtype='float32')
        fn = "points_{}_{}_seed{}.npz".format(self.args.model_name, self.args.data_name, self.args.seed)

        points_path = os.path.join(self.args.root_path, "points")
        os.makedirs(points_path, exist_ok=True)
        fn = os.path.join(points_path, fn)
        np.savez_compressed(fn, time=save_time, space=save_space, label=save_label, predict=save_predict)

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)


        # [ Metrics on T ]
        mae_list = list()  # List length is same as args.predict_time_step=12
        rmse_list = list()
        mape_list = list()
        for t in range(self.args.predict_time_step):
            pred = y_pred[:, t, :]
            true = y_true[:, t, :]
            mae, rmse, mape = cal_metrics(true, pred)
            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)
            self.logger.log_and_print("{:02d} min, MAE: {:.3f}, RMSE: {:.3f}, MAPE: {:.3f}%".format(
                (t+1)*5, statistics.mean(mae_list), statistics.mean(rmse_list), statistics.mean(mape_list) * 100
            ))