import itertools
import os
from sklearn.manifold import TSNE
from torch_geometric.utils import negative_sampling
import baseline_fairdrop
from os.path import join, dirname, realpath
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch_geometric.utils import dropout_adj, convert
from baseline_fairdrop import *
import torch.optim as optim
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.utils import train_test_split_edges
from utils import *
from itertools import product

torch.cuda.empty_cache()

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



config = dict(
    learning_rate=0.01, walk_length=30, walks_per_node=10, p=0.50, q=0.75, delta=0.3
)

dataset = 'bail' 
nosens = False


# Load credit_scoring dataset
if dataset == 'credit':
    sens_attr = "Age"  # column number after feature process is 1
    sens_idx = 1
    predict_attr = 'NoDefaultNextMonth'
    label_number = 6000
    path_credit = "./dataset/credit"
    adj, features, features_nosens, labels, idx_train, idx_val, idx_test, sens = load_credit(dataset, sens_idx, sens_attr,
                                                                            predict_attr, path=path_credit,
                                                                            label_number=label_number
                                                                            )
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    norm_features_nosens = feature_norm(features_nosens)
    features = norm_features # all features
    features_nosens = norm_features_nosens # all features except sensitive one




# Load german dataset
elif dataset == 'german':
    sens_attr = "Gender"  # column number after feature process is 0
    sens_idx = 0
    predict_attr = "GoodCustomer"
    label_number = 100
    path_german = "./dataset/german"
    adj, features, features_nosens, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_idx, sens_attr,
                                                                            predict_attr, path=path_german,
                                                                            label_number=label_number,
                                                                            )



# Load bail dataset
elif dataset == 'bail':
    sens_attr = "WHITE"  # column number after feature process is 0
    sens_idx = 0
    predict_attr = "RECID"
    label_number = 1000
    path_bail = "./dataset/bail"
    adj, features, features_nosens, labels, idx_train, idx_val, idx_test, sens = load_bail(dataset, sens_idx, sens_attr, 
                                                                            predict_attr, path=path_bail,
                                                                            label_number=label_number,
                                                                            )
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    norm_features_nosens = feature_norm(features_nosens)
    features = norm_features # all features
    features_nosens = norm_features_nosens # all features except sensitive one

#load pokec dataset

elif dataset == 'pokec_z' or dataset == "pokec_n":
    sens_idx = 3	    
    sens_attr = "region"  # column number after feature process is 3
    predict_attr = "I_am_working_in_field"
    label_number = 10000
    # sens_number = args.sens_number
    path_pokec="./dataset/pokec/"

    adj, features, features_nosens, labels, idx_train, idx_val, idx_test, sens = load_pokec(dataset, sens_idx, sens_attr, 
                                                                            predict_attr, path=path_pokec,
                                                                            label_number=label_number,)
    norm_features = feature_norm(features)
    norm_features[:, sens_idx] = features[:, sens_idx]
    norm_features_nosens = feature_norm(features_nosens)
    features = norm_features # all features
    features_nosens = norm_features_nosens # all features except sensitive one

else:
    print('Invalid dataset name!!')

edges = convert.from_scipy_sparse_matrix(adj)[0]
#edges = torch.transpose(edges, 0, 1)


test_seeds = [0,1,2]


sensitive = sens.type(torch.LongTensor)





#deltas = [0.35,0.4] # german
#deltas = [0.47, 0.499] # pokec, credit
#deltas = [0.25, 0.4] # bail

#deltas = [0.35,0.4,0.45,0.49]
deltas = [0.47, 0.499]
Y = torch.LongTensor(sensitive)#.to(device)
Y_aux = (Y[edges[0, :]] != Y[edges[1, :]]).to(device)
        
for delta in deltas:
    acc_auc = []
    fairness = []
    torch.cuda.empty_cache()
    for seed, iter in list(product(test_seeds, [0,1,2])):
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        #delta = combo[1]
        print("delta:", delta)
        #random_seed = combo[0]
        print("random seed: ", seed)
        #data = norm_features
        #data.train_mask = data.val_mask = data.test_mask = data.y = None
        #data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
        #data = data.to(device)

        num_classes = labels.unique().shape[0]-1
        #N = len()


        epochs = 1000

        if dataset == 'german':
            epochs = 2500

        
        if nosens:
            model = GCN(features_nosens.shape[1], num_classes).to(device)
        else:
            model = GCN(features.shape[1], num_classes).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)



        randomization = (
            torch.FloatTensor(epochs, Y_aux.size(0)).uniform_() < 0.5 + delta
        ).to(device)
        #print("[homo, etero] edges before: \t", torch.bincount(Y_aux.long()), "\n")


        best_val_perf = test_perf = 0
        
        if nosens:
            features_nosens = features_nosens.to(device)
        else:
            features = features.to(device)

        edges = edges.to(device)
        for epoch in range(1, epochs):
            # TRAINING    
            # neg_edges_tr = negative_sampling(
            #     edge_index=data.train_pos_edge_index,
            #     num_nodes=N,
            #     num_neg_samples=data.train_pos_edge_index.size(1) // 2,
            # ).to(device)


            keep = torch.where(randomization[epoch], Y_aux, ~Y_aux)
            
            model.train()
            optimizer.zero_grad()


            #z = model.encode(data.x, data.train_pos_edge_index[:, keep])
            #link_logits, _ = model.decode(
            #    z, data.train_pos_edge_index[:, keep], neg_edges_tr
            #)
            new_edges = edges[:,keep]

            # Y_aux1 = (Y[new_edges[0, :]] != Y[new_edges[1, :]]).to(device)  
            # print("[homo, etero] edges after: \t", torch.bincount(Y_aux1.long()), "\n")
            if nosens:
                output = model(features_nosens, new_edges)
            else:
                output = model(features, new_edges)

            
            tr_labels = labels[idx_train].unsqueeze(1).float().to(device)


            loss = F.binary_cross_entropy_with_logits(output[idx_train], tr_labels)
            loss.backward()
            optimizer.step()

            # EVALUATION
            # model.eval()
            # perfs = []
            # for prefix in ["val"]:
            #     #pos_edge_index = data[f"{prefix}_pos_edge_index"]
            #     #neg_edge_index = data[f"{prefix}_neg_edge_index"]
            #     with torch.no_grad():
            #         if nosens:
            #             output = model(features_nosens, new_edges)
            #         else:
            #             output = model(features, new_edges)
            #     auc = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
            #     perfs.append(auc)

            # val_perf = perfs[0]
            # if val_perf > best_val_perf:
            #     best_val_perf = val_perf

            # if epoch%100==0:
            #     log = "Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}"
            #     print(log.format(epoch, loss, best_val_perf, test_perf))
            #if epoch%100==0:
            #     log = "Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}"
            #     print(log.format(epoch, loss, best_val_perf))


        # # FAIRNESS
        # auc = test_perf
        # cut = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        # best_acc = 0
        # best_cut = 0.5
        # for i in cut:
        #     acc = accuracy_score(labels.cpu(), output.detach().cpu().numpy() >= i)
        #     if acc > best_acc:
        #         best_acc = acc
        #         best_cut = i
        # #f = prediction_fairness(
        # #    edge_idx.cpu(), link_labels.cpu(), link_probs.cpu() >= best_cut, Y.cpu()
        # #)
        # acc_auc.append([best_acc * 100, auc * 100])
        #fairness.append([x * 100 for x in f])
        torch.cuda.empty_cache()
        model.eval()
        if nosens:
            output = model(features_nosens, edges)
            #counter_features = features.clone()
            #counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
            torch.cuda.empty_cache()
            #counter_output = model(counter_features.to(device), edges)
            output_preds = (output.squeeze()>0).type_as(labels)
            #counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            #counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])

            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
            #f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
            acc_auc.append([auc_roc_test * 100])
            counterfactual_fairness = 0
            fairness.append([x * 100 for x in [parity, equality, counterfactual_fairness]])
            #fairness.append([x * 100 for x in [parity, equality]])
            
        else:
            output = model(features, edges)
            counter_features = features.clone()
            counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
            torch.cuda.empty_cache()
            counter_output = model(counter_features.to(device), edges)
            output_preds = (output.squeeze()>0).type_as(labels)
            counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])

            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
            #f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
            acc_auc.append([auc_roc_test * 100])
            fairness.append([x * 100 for x in [parity, equality, counterfactual_fairness]])
            #fairness.append([x * 100 for x in [parity, equality]])

        # print report
        print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
        print(f'Parity: {parity} | Equality: {equality}')
        #print(f'F1-score: {f1_s}')
        print(f'CounterFactual Fairness: {counterfactual_fairness}')

        randomization.to('cpu')
        model.to('cpu')
        features_nosens.to('cpu')
        features.to('cpu')
        edges.to('cpu')

    ma = np.mean(np.asarray(acc_auc), axis=0)
    mf = np.mean(np.asarray(fairness), axis=0)

    sa = np.std(np.asarray(acc_auc), axis=0)
    sf = np.std(np.asarray(fairness), axis=0)

    print(f"AUC: {ma[0]:2f} +- {sa[0]:2f}")

    print(f"DP: {mf[0]:2f} +- {sf[0]:2f}")
    print(f"EO: {mf[1]:2f} +- {sf[1]:2f}")
    print(f"CF: {mf[2]:2f} +- {sf[2]:2f}")


    # ma = np.mean(np.asarray(acc_auc), axis=0)
# mf = np.mean(np.asarray(fairness), axis=0)

# sa = np.std(np.asarray(acc_auc), axis=0)
# sf = np.std(np.asarray(fairness), axis=0)

# print(f"ACC: {ma[0]:2f} +- {sa[0]:2f}")
# print(f"AUC: {ma[1]:2f} +- {sa[1]:2f}")

# print(f"DP mix: {mf[0]:2f} +- {sf[0]:2f}")
# print(f"EoP mix: {mf[1]:2f} +- {sf[1]:2f}")
# print(f"DP group: {mf[2]:2f} +- {sf[2]:2f}")
# print(f"EoP group: {mf[3]:2f} +- {sf[3]:2f}")
# print(f"DP sub: {mf[4]:2f} +- {sf[4]:2f}")
# print(f"EoP sub: {mf[5]:2f} +- {sf[5]:2f}")



# num_classes = 2

# N = sensitive.shape[0]

# m = np.random.choice(len(edges), int(len(edges) * 0.8), replace=False)
# tr_mask = np.zeros(len(edges), dtype=bool)
# tr_mask[m] = True
# pos_edges_tr = edges[tr_mask]
# pos_edges_te = edges[~tr_mask]


# pos_edges_te = torch.LongTensor(pos_edges_te.T).to(device)
# neg_edges_te = negative_sampling(
#     edge_index=pos_edges_te, num_nodes=N, num_neg_samples=pos_edges_te.size(1)
# ).to(device)

# pos_edges_tr = torch.LongTensor(pos_edges_tr.T).to(device)
# neg_edges_tr = negative_sampling(
#     edge_index=pos_edges_tr, num_nodes=N, num_neg_samples=pos_edges_tr.size(1)
# ).to(device)


# epochs = 51
# model = Node2Vec(
#     pos_edges_tr,
#     embedding_dim=128,
#     walk_length=config["walk_length"],
#     context_size=10,
#     walks_per_node=config["walks_per_node"],
#     p=config["p"],
#     q=config["q"],
#     num_negative_samples=1,
#     sparse=True,
# ).to(device)

# loader = model.loader(batch_size=64, shuffle=True, num_workers=8)

# optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=config["learning_rate"])

# Y = torch.LongTensor(sensitive).to(device)
# Y_aux = (Y[pos_edges_tr[0, :]] != Y[pos_edges_tr[1, :]]).to(device)
# randomization = (torch.FloatTensor(epochs, Y_aux.size(0)).uniform_() < 0.5 + config["delta"]).to(
#     device
# )


# for epoch in range(1, epochs):

#     loss = train_rn2v_adaptive(
#         model,
#         loader,
#         optimizer,
#         device,
#         pos_edges_tr,
#         Y_aux,
#         randomization[epoch],
#         N,
#     )

# model.eval()
# scaler = preprocessing.StandardScaler()
# XB = scaler.fit_transform(model().detach().cpu())
# print(XB)
# YB = sensitive

# node_rb = emb_fairness(XB, YB)
# print(node_rb)