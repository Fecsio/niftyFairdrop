import regex
from unicodedata import numeric
import pandas as pd


#df0 = pd.read_csv("risultati0.txt", sep='\t', lineterminator='\r', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#df0.loc['mean'] = df0.mean(numeric_only=True)
#df0.loc['stdev'] = df0.std(numeric_only=True)

#df1 = pd.read_csv("risultati1.txt", sep='\t', lineterminator='\r', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
#df1.loc['mean'] = df1.mean(numeric_only=True)
#df1.loc['stdev'] = df1.std(numeric_only=True)

#print(df0.tail(2), "\n")

#print(df1.tail(2), "\n")

#lines = []
#for i in ["-bail-longrun", "-credit-longrun", "-german-longrun", "-pokec_n-longrun"]:
#for i in ["-bail-longrun", "-pokec_n-longrun"]:
#for i in ["-pokec_n-longrun-0.25-0.3-0.35.txt"]:
#for i in ["-german-longrun-nofairdrop-035"]:
#for i in ["-pokec_n-longrun-0.47-0.49-0.499"]:
for i in ["german",'credit','bail']:
    f = i + "output-nifty.txt" 
    with open(f, 'r') as file:
        with open(i+"pulito-nifty.txt", 'w') as fileo:
            for line in file:
                if 'seed' not in line and 'AUROC' not in line and 'drop' not in line and 'delta' not in line and 'time' not in line and "torch" not in line and\
                "Optimization" not in line and "edges" not in line and 'FAIRDROP' not in line and not line.isspace() and '[' not in line:
                    fileo.write(regex.sub('[^0-9.]','\n', line) + '\n')                 

for i in ["german",'credit','bail']:
    f = i + "output.txt"
    with open(f, 'r') as file:
        with open(i+"pulito.txt", 'w') as fileo:
            for line in file:
                if 'seed' not in line and 'AUROC' not in line and 'drop' not in line and 'delta' not in line and 'time' not in line and "torch" not in line and\
                "Optimization" not in line and "edges" not in line and 'FAIRDROP' not in line and not line.isspace() and '[' not in line:
                    fileo.write(regex.sub('[^0-9.]','\n', line) + '\n')                 


              
    #df0 = pd.read_csv(f, sep='\t', lineterminator='\n', names=["metrica", "AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False).dropna() # replace('\n',' ', regex=True)

    #print(df0)

    #df0.to_excel('tabella_risultati' + i + "2.xlsx")

