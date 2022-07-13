from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from utils import *
from torch_geometric.utils import convert


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
nosens = True



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

sensitive = sens
num_classes = labels.unique().shape[0]-1


class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)
     def forward(self, x):
         outputs = (self.linear(x))
         return outputs

epochs = 2500

labels_1 = labels.unsqueeze(1).float()

acc_auc = []
fairness = []   

for i in range(5):
    for seed in test_seeds:
        #if nosens:
        #    X_train, X_test, y_train, y_test = train_test_split(
        #    features_nosens, labels_1, test_size=0.5, random_state=seed)
        #else:
        #    X_train, X_test, y_train, y_test = train_test_split(
        #    features, labels_1, test_size=0.5, random_state=seed)

        X_train, X_test, y_train, y_test = train_test_split(
             features, labels_1, test_size=0.8, random_state=seed)

        s_train = X_train[:, sens_idx]
        s_test = X_test[:, sens_idx]

        if nosens:
            X_train = torch.cat((X_train[:, :sens_idx], X_train[:, sens_idx+1:]),1)
            X_test = torch.cat((X_test[:, :sens_idx], X_test[:, sens_idx+1:]),1)

        if nosens:
            model = LogisticRegression(features_nosens.shape[1], num_classes)
        else:
            model = LogisticRegression(features.shape[1], num_classes)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        optimizer.zero_grad()

        X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
        y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)

        #Training

        losses = []
        losses_test = []
        Iterations = []
        iter = 0

        for epoch in range(epochs+1):
            x = X_train
            y = y_train
            optimizer.zero_grad() 
            outputs = model(X_train)
            loss = criterion(outputs, y) 

            loss.backward() 

            optimizer.step() 

            if epoch%100==0:
                with torch.no_grad():
                    correct_test = 0
                    total_test = 0
                    outputs_test = model(X_test)
                    loss_test = criterion(outputs_test, y_test)
                    auc_roc_test = roc_auc_score(y_test, outputs_test.round().detach().numpy())
                    losses_test.append(loss_test.item())
                    output_preds = (outputs_test.squeeze()>0).type_as(labels)

                    if not nosens:
                        counter_features = X_test.clone()
                        counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
                        counter_output = model(counter_features)
                        counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
                        counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds).sum().item()/len(X_test))

                    else:
                        counterfactual_fairness = 0

                    parity, equality = fair_metric(output_preds.numpy(), y_test.squeeze(1).numpy(), s_test.numpy())

                    auc_roc_train = roc_auc_score(y, outputs.round().detach().numpy())
                    losses.append(loss.item())
                    Iterations.append(epoch)

        acc_auc.append([auc_roc_test * 100])
        fairness.append([x * 100 for x in [parity, equality, counterfactual_fairness]])
    #print(f"Epoch: {epoch}. \nTest - Loss: {loss_test.item()}. AUC: {auc_roc_test}")
    #print(f"AUC: {auc_roc_test}")
    #print(f'Parity: {parity*100} \nEquality: {equality*100}')
    #print(f'CounterFactual Fairness: {counterfactual_fairness*100}')
    #print(f"Train -  Loss: {loss.item()}. AUC: {auc_roc_train}\n")
    #print(losses)

print(np.asarray(fairness))
ma = np.mean(np.asarray(acc_auc), axis=0)
mf = np.mean(np.asarray(fairness), axis=0)

sa = np.std(np.asarray(acc_auc), axis=0)
sf = np.std(np.asarray(fairness), axis=0)

print(f"AUC: {ma[0]:2f} +- {sa[0]:2f}")

print(f"DP: {mf[0]:2f} +- {sf[0]:2f}")
print(f"EO: {mf[1]:2f} +- {sf[1]:2f}")
print(f"CF: {mf[2]:2f} +- {sf[2]:2f}")