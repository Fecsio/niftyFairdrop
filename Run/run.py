#import nifty_sota_gnn
import string
import pandas as pd
import os

# open("german-nifty.txt", "w").close()

# with open("german-nifty.txt", "a") as external_file:
#     for p in [0.01,0.1,0.2]:
#         external_file.write('dr: ' + str(p))
#         open('risultati0.txt', 'w').close()
#         for r in [0,1,2]:
#             for i in range(0,2): 
#                 stringa = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 2500 --model ssf --encoder gcn --dataset german --sim_coeff 0.6' + ' --drop_edge_feature_1 ' + str(p) + ' --drop_edge_feature_2 ' + str(p) + ' --seed ' + str(r) 
#                 os.system(stringa)

#         df0 = pd.read_csv("risultati0.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
#         df0.loc['mean'] = df0.mean(numeric_only=True)
#         df0.loc['stdev'] = df0.std(numeric_only=True)      
#         df0.tail(2).to_csv(external_file, sep='\t')    

# #     external_file.close()

# open("bail-nifty.txt", "w").close()

# with open("bail-nifty.txt", "a") as external_file:
#     for p in [0.01,0.1,0.2]:
#         external_file.write('dr: ' + str(p))
#         open('risultati0.txt', 'w').close()
#         for r in [0,1,2]:
#            for i in range(0,2): 
#                stringa = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6' + ' --drop_edge_feature_1 ' + str(p) + ' --drop_edge_feature_2 ' + str(p) + ' --seed ' + str(r) 
#                os.system(stringa)
#         df0 = pd.read_csv("risultati0.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
#         df0.loc['mean'] = df0.mean(numeric_only=True)
#         df0.loc['stdev'] = df0.std(numeric_only=True)      
#         df0.tail(2).to_csv(external_file, sep='\t')    

#     external_file.close()


open("credit-comb-andfairattr3.txt", "w").close()

with open("credit-comb-andfairattr3.txt", "a") as external_file:
    #for pp in [0.01,0.1,0.2]:
    external_file.write("dr = 0.25" )
    for p in [0.01,0.1,0.2]:
        external_file.write('d_attr_r: ' + str(p))
        for d in [0.47,0.499]:
            open('risultati1.txt', 'w').close()
            open('risultati0.txt', 'w').close()
            external_file.write('delta: ' + str(d))
            for r in [0,1,2]:
               for i in range(0,2): 
                   stringa1 = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.25 --drop_edge_rate_2 0.25 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset credit --sim_coeff 0.6' + ' --drop_feature_rate_1 ' + str(p) + ' --drop_feature_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --fairdrop 1' + " --delta " + str(d) + ' --fairattr 1'
                   os.system(stringa1)
                   stringa = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.25 --drop_edge_rate_2 0.25 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset credit --sim_coeff 0.6' + ' --drop_feature_rate_1 ' + str(p) + ' --drop_feature_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --fairattr 1'
                   if d == 0.47: os.system(stringa)

            df1 = pd.read_csv("risultati1.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
            df1.loc['mean'] = df1.mean(numeric_only=True)
            df1.loc['stdev'] = df1.std(numeric_only=True)
            df0 = pd.read_csv("risultati0.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
            df0.loc['mean'] = df0.mean(numeric_only=True)
            df0.loc['stdev'] = df0.std(numeric_only=True)            
            df1.tail(2).to_csv(external_file, sep='\t') 
            df0.tail(2).to_csv(external_file, sep='\t')   
  

#     external_file.close()
# open("bail-comb-andfairattr2.txt", "w").close()

# with open("bail-comb-andfairattr2.txt", "a") as external_file:
#     #for pp in [0.01,0.1,0.2]:
#     external_file.write("dr = 0.1" )
#     for p in [0.01,0.1,0.2]:
#         external_file.write('d_attr_r: ' + str(p))
#         for d in [0.25,0.4]:
#             open('risultati1.txt', 'w').close()
#             open('risultati0.txt', 'w').close()
#             external_file.write('delta: ' + str(d))
#             for r in [0,1,2]:
#                for i in range(0,2): 
#                    stringa1 = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6' + ' --drop_feature_rate_1 ' + str(p) + ' --drop_feature_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --fairdrop 1' + " --delta " + str(d) + ' --fairattr 1'
#                    os.system(stringa1)
#                    stringa = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset bail --sim_coeff 0.6' + ' --drop_feature_rate_1 ' + str(p) + ' --drop_feature_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --fairattr 1'
#                    if d == 0.25: os.system(stringa)

#             df1 = pd.read_csv("risultati1.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
#             df1.loc['mean'] = df1.mean(numeric_only=True)
#             df1.loc['stdev'] = df1.std(numeric_only=True)
#             df0 = pd.read_csv("risultati0.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
#             df0.loc['mean'] = df0.mean(numeric_only=True)
#             df0.loc['stdev'] = df0.std(numeric_only=True)            
#             df1.tail(2).to_csv(external_file, sep='\t') 
#             df0.tail(2).to_csv(external_file, sep='\t')   
  

# open("german-comb-andfairattr3.txt", "w").close()

# with open("german-comb-andfairattr3.txt", "a") as external_file:
#     #for pp in [0.01,0.1,0.2]:
#     external_file.write("dr = 0.1" )
#     for p in [0.01]:#, 0.1, 0.2]:
#         external_file.write('d_attr_r: ' + str(p))
#         for d in [0.35,0.4]:
#             open('risultati1.txt', 'w').close()
#             open('risultati0.txt', 'w').close()
#             external_file.write('delta: ' + str(d))
#             for r in [0,1,2]:
#                for i in range(0,2): 
#                    stringa1 = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 2500 --model ssf --encoder gcn --dataset german --sim_coeff 0.6' + ' --drop_feature_rate_1 ' + str(p) + ' --drop_feature_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --fairdrop 1' + " --delta " + str(d) + ' --fairattr 1'
#                    os.system(stringa1)
#                    stringa = 'python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 2500 --model ssf --encoder gcn --dataset german --sim_coeff 0.6' + ' --drop_feature_rate_1 ' + str(p) + ' --drop_feature_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --fairattr 1'
#                    if d == 0.35: os.system(stringa)

#             df1 = pd.read_csv("risultati1.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
#             df1.loc['mean'] = df1.mean(numeric_only=True)
#             df1.loc['stdev'] = df1.std(numeric_only=True)
#             df0 = pd.read_csv("risultati0.txt", sep=',', lineterminator='\n', names=["AUROC", "Parity", "Equality", "Counterfactual"], index_col=False)
#             df0.loc['mean'] = df0.mean(numeric_only=True)
#             df0.loc['stdev'] = df0.std(numeric_only=True)            
#             df1.tail(2).to_csv(external_file, sep='\t')   
#             df0.tail(2).to_csv(external_file, sep='\t')   
