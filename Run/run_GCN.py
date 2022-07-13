from cgi import test
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from utils import *
from torch_geometric.utils import convert
from baseline_fairdrop import GCN
import torch
import matplotlib.pyplot as plt
import seaborn as sns

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


test_seeds = [0,1,2]

sensitive = sens.type(torch.LongTensor)
num_classes = labels.unique().shape[0]-1


# def correlation_heatmap(train):
#     correlations = train.corr()

#     fig, ax = plt.subplots(figsize=(30,30))

#     sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
#                 square=True, linewidths=.5, annot=False, cbar_kws={"shrink": .70}, cmap='vlag')
#     for _ in range(2):
#         x = sens_idx
#         w = 1
#         y = 0
#         h = features.shape[1]
#         ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='crimson', lw=1, clip_on=False))
#         ax.add_patch(Rectangle((y, x), h, w, fill=False, edgecolor='crimson', lw=1, clip_on=False))
#         x, y = y, x
#         w, h = h, w
#     plt.savefig("heatmap-labels-" + dataset +  ".png")
#     plt.close()

# #correlation_heatmap(pd.DataFrame(features.numpy()))

# correlation_heatmap(pd.DataFrame(torch.hstack((features, labels.unsqueeze(1))).numpy()))


epochs = 1000

if dataset == 'german':
    epochs = 2500

if nosens:
    features_nosens = features_nosens.to(device)
    model = GCN(features_nosens.shape[1], num_classes)
else:
    features = features.to(device)
    model = GCN(features.shape[1], num_classes)

edges = edges.to(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for seed in test_seeds:
    acc_auc = []
    fairness = []
    for i in range(3): 
        for epoch in range(epochs+1):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.empty_cache()

            model.train()
            optimizer.zero_grad()

            if nosens:
                output = model(features_nosens, edges)
            else:
                output = model(features, edges)

            preds = (output.squeeze()>0).type_as(labels)
            loss_train = torch.nn.functional.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

            auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
            loss_train.backward()
            optimizer.step()

        model.eval()
        if nosens:
            output = model(features_nosens.to(device), edges.to(device))
            output_preds = (output.squeeze()>0).type_as(labels)
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            counterfactual_fairness = 0
            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())
        else:
            output = model(features.to(device), edges.to(device))
            counter_features = features.clone()
            counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
            counter_output = model(counter_features.to(device), edges.to(device))
            output_preds = (output.squeeze()>0).type_as(labels)
            counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
            auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
            counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])
            parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())

        acc_auc.append([auc_roc_test * 100])
        fairness.append([x * 100 for x in [parity, equality, counterfactual_fairness]])


ma = np.mean(np.asarray(acc_auc), axis=0)
mf = np.mean(np.asarray(fairness), axis=0)

sa = np.std(np.asarray(acc_auc), axis=0)
sf = np.std(np.asarray(fairness), axis=0)

print(f"AUC: {ma[0]:2f} +- {sa[0]:2f}")

print(f"DP: {mf[0]:2f} +- {sf[0]:2f}")
print(f"EO: {mf[1]:2f} +- {sf[1]:2f}")
print(f"CF: {mf[2]:2f} +- {sf[2]:2f}")


