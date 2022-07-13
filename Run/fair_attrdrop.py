import torch



def fair_drop_feature(x, drop_prob, sens_idx, related_attrs, related_weights, sens_flag=True):
    # drop_mask = torch.empty(
    #     (x.size(1), ),
    #     dtype=torch.float32,
    #     device=x.device).uniform_(0, 1) < drop_prob

    # x = x.clone()
    # drop_mask[sens_idx] = False

    # x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    drop_mask = torch.empty(
        (x.size(1)),
        dtype=torch.float32,
        device=x.device).uniform_(0,1) < drop_prob
    
    rel_mask = torch.empty(
         (len(related_weights)),
         dtype=torch.float32,
         device=x.device).uniform_(0, 1) 

    
    drop_mask[related_attrs] = rel_mask < torch.FloatTensor([drop_prob + i for i in related_weights]).to(x.device)

    x = x.clone()
    drop_mask[sens_idx] = False

    x[:, drop_mask] += torch.ones(1).normal_(0, 1).to(x.device)

    # Flip sensitive attribute
    if sens_flag:
        x[:, sens_idx] = 1-x[:, sens_idx]

    return x

