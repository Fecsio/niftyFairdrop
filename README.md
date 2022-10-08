## Biased Edge Dropout & Fair Related Features in NIFTY for Fair Graph Representation Learning 

Part of this repository contains source code necessary to reproduce some of the main results in [the paper](https://www.esann.org/sites/default/files/proceedings/2022/ES2022-99.pdf).

Moreover, an extension have been developed to complete my master's degree thesis.

Some running examples can be found in *Run* folder. 
As an example, the file run.py show the running command:

```
python nifty_sota_gnn-fairattr.py --drop_edge_rate_1 0.1 --drop_edge_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 2500 --model ssf --encoder gcn --dataset german --sim_coeff 0.6' + ' --drop_feature_rate_1 ' + str(p) + ' --drop_feature_rate_2 ' + str(p) + ' --seed ' + str(r) + ' --fairdrop 1' + " --delta " + str(d) + ' --fairattr 1
```

where:

- nifty_sota_gnn-fairattr.py is the script that can run the entire solution with Biased Edge Dropout AND Fair Related Features
- drop_edge_rate_1 and 2 is the drop edge rate used by NIFTY (values should be the same)
- drop_feature_rate_1 and 2 same as the previous point but w.r.t. nodes attributes
- delta is the delta hyperparameter as defined in the paper
- fairattr is a boolean indicating wheter using or not the **fair related features** extension
- fairdrop is a boolean indicating wheter using or not the **biased edge dropout** extension

Values for hyperparameters can be found both in the paper and in the [thesis](https://thesis.unipd.it/handle/20.500.12608/32821) 

