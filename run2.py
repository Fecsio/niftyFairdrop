import os
import pandas as pd


open('risultati0.txt', 'w').close()
open('risultati1.txt', 'w').close()
for j in [0,1,2]:
    print("seed: ", j)
    for i in range(0,3): 
           os.system('python nifty_sota_gnn.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6 --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1' + ' --seed ' + str(j))
           os.system('python nifty_sota_gnn.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6 --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --fairdrop 1 --delta 0' + ' --seed ' + str(j))

df0 = pd.read_csv("risultati0.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
df0.loc['mean'] = df0.mean(numeric_only=True)
df0.loc['stdev'] = df0.std(numeric_only=True)

df1 = pd.read_csv("risultati1.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
df1.loc['mean'] = df1.mean(numeric_only=True)
df1.loc['stdev'] = df1.std(numeric_only=True)

print(df0.tail(2))
print(df1.tail(2))