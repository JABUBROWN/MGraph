import torch


# Calculate MAE, RMSE, MAPE of 2 Tensors at Validation and Test stage.
def cal_metrics(true, pred, mask_value=1e-3):
    mae = float(torch.mean(torch.abs(true - pred)))
    rmse = float(torch.sqrt(torch.mean((true - pred) ** 2)))
    mask = torch.gt(true, mask_value)
    true = torch.masked_select(true, mask)
    pred = torch.masked_select(pred, mask)
    mape = float(torch.mean(torch.abs(torch.div((true - pred), true))))

    return mae, rmse, mape
