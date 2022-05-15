#import nifty_sota_gnn
import os
import pandas as pd
import time


#with open("risultati.txt", "w") as external_file:
#    external_file.write('NIFTY standard drop')
#    external_file.close()



#for j in range(0,5):
#    os.system('python media.py')
#    open('risultati0.txt', 'w').close()
#    open('risultati1.txt', 'w').close()
#    for i in range(0,5): 
#        os.system('python nifty_sota_gnn.py --drop_edge_rate_1 0.01 --drop_edge_rate_2 0.01 --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6 --seed 42')
#        os.system('python nifty_sota_gnn.py --drop_edge_rate_1 0.001 --drop_edge_rate_2 0.001 --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6 --seed 42 --fairdrop 1')


#for i in range(0,5):
 #os.system('python nifty_sota_gnn.py --drop_edge_rate_1 0.001 --drop_edge_rate_2 0.001 --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6 --seed 1 --fairdrop 1')



# open("risultatitot-bail-longrun-0.25-0.3-0.35.txt", "w").close()

# with open("risultatitot-bail-longrun-0.25-0.3-0.35.txt", "a") as external_file:
#     for r in [0,1,2]:
#        external_file.write('seed: ' + str(r) + ';\t')
#        for p in [0.25,0.3,0.35]:
#            open('risultati0.txt', 'w').close()
#            open('risultati1.txt', 'w').close()
#            external_file.write(' drop edge pr: ' + str(p) + '\n\t')
#            for d in [0.25, 0.4, 0.49]:
#                open('risultati0.txt', 'w').close()
#                open('risultati1.txt', 'w').close()
#                external_file.write(' delta: ' + str(d) + '\n\t')
#                for i in range(0,4): 
#                    stringa = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) 
#                    stringa1 = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset  bail --sim_coeff 0.6 --fairdrop 1' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --delta ' + str(d)
#                    if d == 0.25:  os.system(stringa) #faccio girare solo una volta
#                    os.system(stringa1)


#                df0 = pd.read_csv("risultati0.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#                df0.loc['mean'] = df0.mean(numeric_only=True)
#                df0.loc['stdev'] = df0.std(numeric_only=True)
#                df1 = pd.read_csv("risultati1.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#                df1.loc['mean'] = df1.mean(numeric_only=True)
#                df1.loc['stdev'] = df1.std(numeric_only=True)

#                df0.tail(2).to_csv(external_file, sep='\t')
#                df1.tail(2).to_csv(external_file, sep='\t')            

#     external_file.close()


# open("risultatitot-pokec_n-longrun-0.47-0.49-0.499.txt", "w").close()


# with open("risultatitot-pokec_n-longrun-0.47-0.49-0.499.txt", "a") as external_file:
#    for r in [0,1,2]:
#        external_file.write('seed: ' + str(r) + ';\t')
#        for p in [0.01,0.1,0.2]:
#            open('risultati0.txt', 'w').close()
#            open('risultati1.txt', 'w').close()
#            external_file.write(' drop edge pr: ' + str(p) + '\n\t')
#            for d in [0.47,0.49,0.499]:
#                open('risultati0.txt', 'w').close()
#                open('risultati1.txt', 'w').close()
#                external_file.write(' delta: ' + str(d) + '\n\t')
#                for i in range(0,4): 
#                    stringa = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset pokec_n --sim_coeff 0.6' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) 
#                    stringa1 = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset  pokec_n --sim_coeff 0.6 --fairdrop 1' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --delta ' + str(d)
#                    if d == 0.47:  os.system(stringa) #faccio girare solo una volta
#                    os.system(stringa1)


#                df0 = pd.read_csv("risultati0.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#                df0.loc['mean'] = df0.mean(numeric_only=True)
#                df0.loc['stdev'] = df0.std(numeric_only=True)

#                df1 = pd.read_csv("risultati1.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#                df1.loc['mean'] = df1.mean(numeric_only=True)
#                df1.loc['stdev'] = df1.std(numeric_only=True)

#                df0.tail(2).to_csv(external_file, sep='\t')
#                df1.tail(2).to_csv(external_file, sep='\t')            

#    external_file.close()

# open("risultatitot-german-longrun-nofairdrop-035.txt", "w").close()


# with open("risultatitot-german-longrun-nofairdrop-035.txt", "a") as external_file:
#     for r in [0,1,2]:
#         external_file.write('seed: ' + str(r) + ';\t')
#         for p in [0.01,0.1,0.2, 0.25]:
#             open('risultati0.txt', 'w').close()
#             open('risultati1.txt', 'w').close()
#             external_file.write(' drop edge pr: ' + str(p) + '\n\t')
#             for d in [0.35, 0.4, 0.49]:
#                 open('risultati0.txt', 'w').close()
#                 open('risultati1.txt', 'w').close()
#                 external_file.write(' delta: ' + str(d) + '\n\t')
#                 for i in range(0,4): 
#                     stringa = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 2500 --model ssf --encoder gcn --dataset german --sim_coeff 0.6' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) 
#                     stringa1 = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 2500 --model ssf --encoder gcn --dataset  german --sim_coeff 0.6 --fairdrop 1' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --delta ' + str(d)
#                     if d == 0.35:  os.system(stringa) #faccio girare solo una volta
#                     #os.system(stringa1)


#                 df0 = pd.read_csv("risultati0.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#                 df0.loc['mean'] = df0.mean(numeric_only=True)
#                 df0.loc['stdev'] = df0.std(numeric_only=True)

#                 df1 = pd.read_csv("risultati1.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#                 df1.loc['mean'] = df1.mean(numeric_only=True)
#                 df1.loc['stdev'] = df1.std(numeric_only=True)

#                 df0.tail(2).to_csv(external_file, sep='\t')
#                 df1.tail(2).to_csv(external_file, sep='\t')            

#     external_file.close()

# open("risultatitot-german-longrun-nofairdrop-035.txt", "w").close()

open("risultatitot-credit-longrun-001.txt", "w").close()

with open("risultatitot-credit-longrun-001.txt", "a") as external_file:
    for r in [0,1,2]:
        external_file.write('seed: ' + str(r) + ';\t')
        for p in [0.01]:
            open('risultati0.txt', 'w').close()
            open('risultati1.txt', 'w').close()
            external_file.write(' drop edge pr: ' + str(p) + '\n\t')
            for d in [0.47, 0.499]:
                open('risultati0.txt', 'w').close()
                open('risultati1.txt', 'w').close()
                external_file.write(' delta: ' + str(d) + '\n\t')
                for i in range(0,4): 
                    stringa = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset credit --sim_coeff 0.6' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) 
                    stringa1 = 'python nifty_sota_gnn.py --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset  credit --sim_coeff 0.6 --fairdrop 1' + ' --drop_edge_rate_1 ' + str(p) + ' --drop_edge_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --delta ' + str(d)
                    #if d == 0.47 and p == 0.3:  os.system(stringa) #faccio girare solo una volta
                    os.system(stringa1)


                df0 = pd.read_csv("risultati0.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
                df0.loc['mean'] = df0.mean(numeric_only=True)
                df0.loc['stdev'] = df0.std(numeric_only=True)

                df1 = pd.read_csv("risultati1.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
                df1.loc['mean'] = df1.mean(numeric_only=True)
                df1.loc['stdev'] = df1.std(numeric_only=True)

                df0.tail(2).to_csv(external_file, sep='\t')
                df1.tail(2).to_csv(external_file, sep='\t')            

    external_file.close()


