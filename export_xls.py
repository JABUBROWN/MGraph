import os
import pandas as pd
from statistics import mean

# Export specified experiment SEED 1 ~ 10 to Excel file.

DATA = "PeMS08"
MODEL = "GMAN"

mae_list = [[], [], [], []]
rmse_list = [[], [], [], []]
mape_list = [[], [], [], []]

for seed in range(10):
    fn = "{}_{}_seed{}.txt".format(MODEL, DATA, seed+1)
    with open(os.path.join('log', fn)) as f:
        idx = 0
        for line in f.readlines():
            if 'min' not in line:
                continue
            idx += 1
            if idx > 12:
                print("[ERROR] Redundant experimental records in {}".format(fn))
            if idx % 3 == 0:  # 15min, 30min, 45min, 60min
                parts = line.split(',')
                mae = float(parts[1].split(':')[1].strip())
                rmse = float(parts[2].split(':')[1].strip())
                mape = float(parts[3].split(':')[1].strip()[:-1]) / 100
                mae_list[idx // 3 - 1].append(mae)
                rmse_list[idx // 3 - 1].append(rmse)
                mape_list[idx // 3 - 1].append(mape)

dfData = {'15mae': mae_list[0], '15rmse': rmse_list[0], '15mape': mape_list[0],
          '30mae': mae_list[1], '30rmse': rmse_list[1], '30mape': mape_list[1],
          '45mae': mae_list[2], '45rmse': rmse_list[2], '45mape': mape_list[2],
          '60mae': mae_list[3], '60rmse': rmse_list[3], '60mape': mape_list[3]}
df = pd.DataFrame(dfData)
df.to_excel("experiments_{}_{}.xlsx".format(MODEL, DATA), index=False)
