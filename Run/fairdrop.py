from torch import Tensor, cat, stack, bernoulli, float, bool , where, bincount, multinomial, full, randperm, from_numpy
from typing import Tuple
import numpy as np
from random import randint

def find_intersect_2(t1, t2):
    combined = cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return intersection

def filter_fair_adj(row, col, mask):
    return row[mask], col[mask]

def fairdrop_adj(
    edge_index: Tensor,
    Y,
    Y_aux,
    p_homo,
    p,
    device
) -> Tensor:

    row, col = edge_index

    mask = edge_index.new_full((row.size(0), ), 1-p, dtype=float)
    mask = bernoulli(mask).to(bool)

    #Y_aux1 = (Y[edge_index_tmp[0, :]] != Y[edge_index_tmp[1, :]]).to(device) 

    row_tmp_rev, col_tmp_rev = filter_fair_adj(row, col, ~mask)
    diffs = stack([row_tmp_rev, col_tmp_rev], dim=0)
    
    Y_aux1 = (Y[diffs[0, :]] != Y[diffs[1, :]]).to(device) 

    omo, etero = bincount(Y_aux1.long())
    omo = omo.item()
    etero = etero.item()

    perc = (omo/(omo+etero))


    if perc < p_homo:
        diff = (p_homo - perc) * p*edge_index.size(1) 
        diff = int(diff) # numero di edge omofili da togliere

        b = mask == True # indici tenuti
        bb = Y_aux == True # indici eterofili
            
        # ~b indici tolti
        # ~bb indici omofili

        ind_togli = find_intersect_2(b.nonzero(), (~bb).nonzero()) # indici toglibli
        ind_aggiungi = find_intersect_2((~b).nonzero(), bb.nonzero()) # indici aggiungibili
        
        if len(ind_togli) < diff or len(ind_aggiungi) < diff: # controlla che ci siano abbastanza edges da togliere/aggiungere
            diff = min(len(ind_togli), len(ind_aggiungi))


        r_togli = randperm(len(ind_togli))[:diff]
        r_agg = randperm(len(ind_aggiungi))[:diff]

        ind_togli = ind_togli[r_togli]
        ind_aggiungi = ind_aggiungi[r_agg]

        mask[ind_aggiungi[:]] = True
        mask[ind_togli[:]] = False

        row, col = filter_fair_adj(row, col, mask)
        edge_index = stack([row, col], dim=0)

    else:
        row_tmp, col_tmp = filter_fair_adj(row, col, mask)
        edge_index_tmp = stack([row_tmp, col_tmp], dim=0)
        edge_index = edge_index_tmp
    
    return edge_index


