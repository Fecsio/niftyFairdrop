#%%
from ast import arg
import dgl
import ipdb
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from utils import *
from models import *
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from aif360.sklearn.metrics import consistency_score as cs
from aif360.sklearn.metrics import generalized_entropy_error as gee
from fair_attrdrop import fair_drop_feature



def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


def ssf_validation(model, x_1, edge_index_1, x_2, edge_index_2, y):
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # projector
    p1 = model.projection(z1)
    p2 = model.projection(z2)

    # predictor
    h1 = model.prediction(p1)
    h2 = model.prediction(p2)

    l1 = model.D(h1[idx_val], p2[idx_val])/2
    l2 = model.D(h2[idx_val], p1[idx_val])/2
    sim_loss = args.sim_coeff*(l1+l2)

    # classifier
    c1 = model.classifier(z1)
    c2 = model.classifier(z2)

    # Binary Cross-Entropy
    l3 = F.binary_cross_entropy_with_logits(c1[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2
    l4 = F.binary_cross_entropy_with_logits(c2[idx_val], y[idx_val].unsqueeze(1).float().to(device))/2

    return sim_loss, l3+l4


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--proj_hidden', type=int, default=16,
                    help='Number of hidden units in the projection layer of encoder.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--drop_edge_rate_1', type=float, default=0.1,
                    help='drop edge for first augmented graph')
parser.add_argument('--drop_edge_rate_2', type=float, default=0.1,
                    help='drop edge for second augmented graph')
parser.add_argument('--drop_feature_rate_1', type=float, default=0.1,
                    help='drop feature for first augmented graph')
parser.add_argument('--drop_feature_rate_2', type=float, default=0.1,
                    help='drop feature for second augmented graph')
parser.add_argument('--sim_coeff', type=float, default=0.5,
                    help='regularization coeff for the self-supervised task')
parser.add_argument('--dataset', type=str, default='loan',
                    choices=['nba','bail','loan', 'credit', 'german', 'pokec_z', 'pokec_n'])
parser.add_argument("--num_heads", type=int, default=1,
                        help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'sage', 'gin', 'jk', 'infomax', 'ssf', 'rogcn'])
parser.add_argument('--encoder', type=str, default='gcn')
parser.add_argument('--fairdrop', type=int, default=0)
parser.add_argument('--delta', type=float, default=0.25)
parser.add_argument('--fairattr', type=int, default=0)




args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#------------------------ MIE MODIFICHE ------------------------

from fairdrop import fairdrop_adj

delta = args.delta
print("delta: " + str(0.5 + delta))


print("dr:" + str(args.drop_edge_rate_1))
print("seed:" + str(args.seed))
print("fairattr: " + str(args.fairattr))
print("dr_attr_r: " + str(args.drop_feature_rate_1))


acc_auc = []
fairness = []

#---------------------------------------------------------------



# Load data
# print(args.dataset)
dataset = args.dataset

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


edge_index = convert.from_scipy_sparse_matrix(adj)[0]
#print(features[:,sens_idx])
# print(len(idx_train))
# print(len(idx_test))
# print(len(idx_val))

#print(torch.bincount(sens.long()))

def correlation_matrix(train, sens):
    correlations = train.corr('spearman')
    return correlations[sens]

init_corr = correlation_matrix(pd.DataFrame(features.numpy()), sens_idx).abs()

index = init_corr.index
related_attrs = index[init_corr.iloc[:] >= 0.09]
related_weights = init_corr[related_attrs].drop(sens_idx).tolist()
related_attrs = related_attrs.drop(sens_idx).tolist()

print(related_attrs)

weightSum = 0.3
beta = 0.5


#%%    
# Model and optimizer
num_class = labels.unique().shape[0]-1
if args.model == 'gcn':
	model = GCN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

elif args.model == 'sage':
	model = SAGE(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

elif args.model == 'gin':
	model = GIN(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

elif args.model == 'jk':
	model = JK(nfeat=features.shape[1],
	            nhid=args.hidden,
	            nclass=num_class,
	            dropout=args.dropout)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

elif args.model == 'infomax':
	enc_dgi = Encoder_DGI(nfeat=features.shape[1], nhid=args.hidden)
	enc_cls = Encoder_CLS(nhid=args.hidden, nclass=num_class)
	model = GraphInfoMax(enc_dgi=enc_dgi, enc_cls=enc_cls)
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)

elif args.model == 'rogcn':
	model = RobustGCN(nnodes=adj.shape[0], nfeat=features.shape[1], nhid=args.hidden, nclass=num_class, dropout=args.dropout, device=device, seed=args.seed)

elif args.model == 'ssf':
    encoder = Encoder(in_channels=features.shape[1], out_channels=args.hidden, base_model=args.encoder).to(device)	
    model = SSF(encoder=encoder, num_hidden=args.hidden, num_proj_hidden=args.proj_hidden, sim_coeff=args.sim_coeff, nclass=num_class).to(device)
   
    ## ------ MIE MODIFICHE -------------------------------------------
    Y = features.to(device)[:, sens_idx]
    Y_aux = (Y[edge_index[0, :]] != Y[edge_index[1, :]]).to(device)   
    print("[homo, etero] edges before: \t", torch.bincount(Y_aux.long()), "\n")

    if(args.fairdrop):

        print("FAIRDROP\n")
        #r = (torch.FloatTensor(Y_aux.size(0)).uniform_() < 0.5 + delta).to(device)

        print("Number of edges before: \t", edge_index.size(1), "\n")
        val_edge_index_1 = fairdrop_adj(edge_index.to(device), Y=Y, Y_aux=Y_aux, p_homo=0.5+delta, p=args.drop_edge_rate_1, device=device)
        val_edge_index_2 = fairdrop_adj(edge_index.to(device), Y=Y, Y_aux=Y_aux, p_homo=0.5+delta, p=args.drop_edge_rate_1, device=device)
        print("Number of edges after: \t", val_edge_index_1.size(1), "\n")


    else:
        print("NO FAIRDROP\n")
    ## ---------------------------------------------------------------
        print("Number of edges before: \t", edge_index.size(1), "\n")
        val_edge_index_1 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_1)[0]
        val_edge_index_2 = dropout_adj(edge_index.to(device), p=args.drop_edge_rate_2)[0]
        print("Number of edges after: \t", val_edge_index_1.size(1), "\n")

    ## ------ MIE MODIFICHE -------------------------------------------

    Y_aux1 = (Y[val_edge_index_1[0, :]] != Y[val_edge_index_1[1, :]]).to(device)  
    print("[homo, etero] edges after: \t", torch.bincount(Y_aux1.long()), "\n")

    ## ---------------------------------------------------------------  

    val_x_1 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, sens_flag=False)
    val_x_2 = drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx)
    
    #val_x_1 = fair_drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, related_attrs, related_weights, sens_flag=False)
    #val_x_2 = fair_drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, related_attrs, related_weights)
    
    
    par_1 = list(model.encoder.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.fc4.parameters())
    par_2 = list(model.c1.parameters()) + list(model.encoder.parameters())
    optimizer_1 = optim.Adam(par_1, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_2 = optim.Adam(par_2, lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)


# Train model
t_total = time.time()
best_loss = 100
best_acc = 0
features = features.to(device)
edge_index = edge_index.to(device)
labels = labels.to(device)

if args.model == 'rogcn':
    model.fit(features, adj, labels, idx_train, idx_val=idx_val, idx_test=idx_test, verbose=True, attention=False, train_iters=args.epochs)

## ---------------MIE MODIFICHE-----------------------------------------------------------------------

#if(args.fairdrop):
#    Y = features.to(device)[:, sens_idx]
#    Y_aux = (Y[edge_index[0, :]] != Y[edge_index[1, :]]).to(device) 

    #r = (torch.FloatTensor(args.epochs, Y_aux.size(0)).uniform_() < 0.5 + delta).to(
    #    device
    #)

##-----------------------------------------------------------------------------------------------------

for epoch in range(args.epochs+1):
    t = time.time()

    if args.model in ['gcn', 'sage', 'gin', 'jk', 'infomax']:
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)

        # Binary Cross-Entropy  
        preds = (output.squeeze()>0).type_as(labels)
        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))

        auc_roc_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        model.eval()
        output = model(features, edge_index)

        # Binary Cross-Entropy
        preds = (output.squeeze()>0).type_as(labels)
        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float().to(device))

        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])
        f1_val = f1_score(labels[idx_val].cpu().numpy(), preds[idx_val].cpu().numpy())

        #if epoch % 100 == 0:
        #    print(f"[Train] Epoch {epoch}:train_loss: {loss_train.item():.4f} | train_auc_roc: {auc_roc_train:.4f} | val_loss: {loss_val.item():.4f} | val_auc_roc: {auc_roc_val:.4f}")

        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            torch.save(model.state_dict(), 'weights_vanilla.pt')

    elif args.model == 'ssf':
        sim_loss = 0
        cl_loss = 0
        rep = 1
        for _ in range(rep):
            model.train()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()            

            ## ---------------MIE MODIFICHE---------------------------------------------------------------------------------
            
            emb = model(features, edge_index)
            output = model.predict(emb)
            p_y = (output.squeeze()>0).type_as(labels).float()[idx_train]

            if args.fairdrop:
                edge_index_1 = fairdrop_adj(edge_index.to(device), Y=Y, Y_aux=Y_aux, p=args.drop_edge_rate_1, p_homo=0.5+delta, device=device)
                edge_index_2 = fairdrop_adj(edge_index.to(device), Y=Y, Y_aux=Y_aux, p=args.drop_edge_rate_2, p_homo=0.5+delta, device=device)
            else:
            ## --------------------------------------------------------------------------------------------------------------
                edge_index_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
                edge_index_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]

            x_1 = drop_feature(features, args.drop_feature_rate_2, sens_idx, sens_flag=False) ## Node perturbation
            x_2 = drop_feature(features, args.drop_feature_rate_2, sens_idx) ## 'Counterfactual' perturbation
            
            #x_1 = fair_drop_feature(features, args.drop_feature_rate_2, sens_idx, related_attrs, related_weights, sens_flag=False)
            #x_2 = fair_drop_feature(features, args.drop_feature_rate_2, sens_idx, related_attrs, related_weights)


            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)

            # projector
            p1 = model.projection(z1)
            p2 = model.projection(z2)

            # predictor
            h1 = model.prediction(p1)
            h2 = model.prediction(p2)

            l1 = model.D(h1[idx_train], p2[idx_train])/2
            l2 = model.D(h2[idx_train], p1[idx_train])/2
            sim_loss += args.sim_coeff*(l1+l2)

        (sim_loss/rep).backward()
        optimizer_1.step()

        # classifier
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        c1 = model.classifier(z1)
        c2 = model.classifier(z2)

        # Binary Cross-Entropy    
        l3 = F.binary_cross_entropy_with_logits(c1[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2
        l4 = F.binary_cross_entropy_with_logits(c2[idx_train], labels[idx_train].unsqueeze(1).float().to(device))/2

        cl_loss = (1-args.sim_coeff)*(l3+l4)
        
        ## ---------------MIE MODIFICHE---------------------------------------------------------------------------------
        
        # for related_attr, related_weight in zip(related_attrs, related_weights):
        #         selected_column = related_attrs

        #         # print(features[idx_train][:,selected_column].shape)
        #         # print(features[idx_train][:,selected_column].reshape(1,features[idx_train][:,selected_column].shape[0],-1).shape)
        #         # print((features[idx_train][:,selected_column]).float())
        #         # #print(features[idx_train][:,selected_column].reshape(1,features[idx_train][:,selected_column].shape[0],-1).float() - features[idx_train][:,selected_column].float().mean(dim=0))
        #         #print(features[idx_train][:,selected_column].reshape(1,features[idx_train][:,selected_column].shape[0],-1).float() - features[idx_train][:,selected_column].float().mean(dim=0))
        #         #print((p_y-p_y.mean(dim=0)).unsqueeze(1))
        #         #print(features[idx_train][:,selected_column].float().mean(dim=0))

        #         cor_loss1 = torch.mul(features[idx_train][:,selected_column].reshape(1,features[idx_train][:,selected_column].shape[0],-1).float() - features[idx_train][:,selected_column].float().mean(dim=0), (p_y-p_y.mean(dim=0)).unsqueeze(1))
        #         cor_loss2 = torch.mean(cor_loss1)
        #         cor_loss3 = torch.abs(cor_loss2)
        #         cor_loss3 = torch.sum(cor_loss2)
        #         cor_loss = cor_loss3
                
        #         #if epoch % 100 == 0:
        #         #    print('classification loss: {}, feature correlation loss: {}'.format(cl_loss.item(), cor_loss.item()))
        #         cl_loss = cl_loss + cor_loss*related_weight*weightSum
        ## --------------------------------------------------------------------------------------------------------------

        cl_loss.backward()
        optimizer_2.step()
        loss = (sim_loss/rep + cl_loss)
        
        ## ---------------MIE MODIFICHE---------------------------------------------------------------------------------

        for iter in range(1):
            with torch.no_grad():
                emb = model(features, edge_index)
                output = model.predict(emb)
                p_y = (output.squeeze()>0).type_as(labels).float()[idx_train]

                cor_losses = []
                for related_attr in related_attrs:
                    selected_column = related_attrs
                    cor_loss1 = torch.mul(features[idx_train][:,selected_column].reshape(1,features[idx_train][:,selected_column].shape[0],-1).float() - features[idx_train][:,selected_column].float().mean(dim=0), (p_y-p_y.mean(dim=0)).unsqueeze(1))
                    cor_loss2 = torch.mean(cor_loss1)
                    cor_loss3 = torch.abs(cor_loss2)
                    cor_loss3 = torch.sum(cor_loss2)
                    cor_loss = cor_loss3
                    
                    cor_losses.append(cor_loss.item())

                cor_losses = np.array(cor_losses)

                cor_order = np.argsort(cor_losses)

                #compute -v. represent it as v.
                beta = beta
                v = cor_losses[cor_order[0]]+ 2*beta
                cor_sum = cor_losses[cor_order[0]]
                l=1
                for i in range(cor_order.shape[0]-1):
                    if cor_losses[cor_order[i+1]] < v:
                        cor_sum = cor_sum + cor_losses[cor_order[i+1]]
                        v = (cor_sum+2*beta)/(i+2)
                        l = l+1
                    else:
                        break
                
                #compute lambda
                for i in range(cor_order.shape[0]):
                    if i < l:
                        related_weights[cor_order[i]] = (v-cor_losses[cor_order[i]])/(2*beta)
                    else:
                        related_weights[cor_order[i]] = 0        
            ## --------------------------------------------------------------------------------------------------------------

        # Validation
        model.eval()
        val_x_1 = fair_drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, related_attrs, related_weights, sens_flag=False)
        val_x_2 = fair_drop_feature(features.to(device), args.drop_feature_rate_2, sens_idx, related_attrs, related_weights)
        val_s_loss, val_c_loss = ssf_validation(model, val_x_1, val_edge_index_1, val_x_2, val_edge_index_2, labels)
        emb = model(val_x_1, val_edge_index_1)
        output = model.predict(emb)
        preds = (output.squeeze()>0).type_as(labels)
        auc_roc_val = roc_auc_score(labels.cpu().numpy()[idx_val], output.detach().cpu().numpy()[idx_val])

        #if epoch % 100 == 0:
        #    print(f"[Train] Epoch {epoch}:train_s_loss: {(sim_loss/rep):.4f} | train_c_loss: {cl_loss:.4f} | val_s_loss: {val_s_loss:.4f} | val_c_loss: {val_c_loss:.4f} | val_auc_roc: {auc_roc_val:.4f}")

        if (val_c_loss + val_s_loss) < best_loss:
            #print(f'{epoch} | {val_s_loss:.4f} | {val_c_loss:.4f}')
            best_loss = val_c_loss + val_s_loss
            torch.save(model.state_dict(), f'weights_ssf_{args.encoder}.pt')

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if args.model in ['gcn', 'sage', 'gin', 'jk', 'infomax']:
    model.load_state_dict(torch.load('weights_vanilla.pt'))
    model.eval()
    output = model(features.to(device), edge_index.to(device))
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model(counter_features.to(device), edge_index.to(device))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model(noisy_features.to(device), edge_index.to(device))

elif args.model == 'rogcn':
    model.load_state_dict(torch.load(f'weights_rogcn_{args.seed}.pt'))
    model.eval()
    model = model.to('cpu')
    output = model.predict(features.to('cpu'))
    counter_features = features.to('cpu').clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(counter_features.to('cpu'))
    noisy_features = features.clone().to('cpu') + torch.ones(features.shape).normal_(0, 1).to('cpu')
    noisy_output = model.predict(noisy_features)

else:
    model.load_state_dict(torch.load(f'weights_ssf_{args.encoder}.pt'))
    model.eval()
    emb = model(features.to(device), edge_index.to(device))
    output = model.predict(emb)
    counter_features = features.clone()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    counter_output = model.predict(model(counter_features.to(device), edge_index.to(device)))
    noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
    noisy_output = model.predict(model(noisy_features.to(device), edge_index.to(device)))
    print(output.size())

# Report
output_preds = (output.squeeze()>0).type_as(labels)
counter_output_preds = (counter_output.squeeze()>0).type_as(labels)
auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
counterfactual_fairness = 1 - (output_preds.eq(counter_output_preds)[idx_test].sum().item()/idx_test.shape[0])

parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].numpy())


# print report
print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
print(f'Parity: {parity} | Equality: {equality}')
#print(f'F1-score: {f1_s}')
print(f'CounterFactual Fairness: {counterfactual_fairness}')
#print(f'Robustness Score: {robustness_score}')

# ------------ MIE MODIFICHE---------------------

with open("risultati"+str(args.fairdrop)+".txt", "a") as external_file:
    external_file.write(f'{auc_roc_test*100},')
    external_file.write(f'{parity*100},')
    external_file.write(f'{equality*100},')
    external_file.write(f'{counterfactual_fairness*100}\n')
    external_file.close()

# ----------------------------------------------------