from matplotlib import pyplot as plt
import pandas as pd
import re
from matplotlib.ticker import PercentFormatter


def makecloud(title, df, parity = True):
    
    if parity: s = "Parity"
    else: s = "Equality"

    for drop_r in [0,1,2]:
        x = df.iloc[0+drop_r*16:8+drop_r*16:2]['AUROC'].reset_index(drop=True)
        y = df.iloc[0+drop_r*16:8+drop_r*16:2][s].reset_index(drop=True)

        x_1 = df.iloc[1+drop_r*16:9+drop_r*16:2]['AUROC'].reset_index(drop=True)
        y_1 = df.iloc[1+drop_r*16:9+drop_r*16:2][s].reset_index(drop=True)
    
    # print("\n FAIRDROP 0.75 \n")

    # print(x_1)
    # print(y_1)
    
        x_2 = df.iloc[8+drop_r*16:12+drop_r*16]['AUROC'].reset_index(drop=True)
        y_2 = df.iloc[8+drop_r*16:12+drop_r*16][s].reset_index(drop=True)
    
    # print("\n FAIRDROP 0.90 \n")

    # print(x_2)
    # print(y_2)

        x_3 = df.iloc[12+drop_r*16:16+drop_r*16]['AUROC'].reset_index(drop=True)
        y_3 = df.iloc[12+drop_r*16:16+drop_r*16][s].reset_index(drop=True)
    
    #print("\n ~~~~~~~~~~~~~~~~", drop_r, "~~~~~~~~~~~~~~~~~~ì\n")
        for seed in [1,2]:
            x = x.append(df.iloc[0+48*seed+drop_r*16:8+48*seed+drop_r*16:2]['AUROC'].reset_index(drop=True))
            y = y.append(df.iloc[0+48*seed+drop_r*16:8+48*seed+drop_r*16:2][s].reset_index(drop=True))

            x_1 = x_1.append(df.iloc[1+48*seed+drop_r*16:9+48*seed+drop_r*16:2]['AUROC'].reset_index(drop=True))
            y_1 = y_1.append(df.iloc[1+48*seed+drop_r*16:9+48*seed+drop_r*16:2][s].reset_index(drop=True))
    
    # print("\n FAIRDROP 0.75 \n")

    # print(x_1)
    # print(y_1)
    
            x_2 = x_2.append(df.iloc[8+48*seed+drop_r*16:12+48*seed+drop_r*16]['AUROC'].reset_index(drop=True))
            y_2 = y_2.append(df.iloc[8+48*seed+drop_r*16:12+48*seed+drop_r*16][s].reset_index(drop=True))
    
    # print("\n FAIRDROP 0.90 \n")

    # print(x_2)
    # print(y_2)

            x_3 = x_3.append(df.iloc[12+48*seed+drop_r*16:16+48*seed+drop_r*16]['AUROC'].reset_index(drop=True))
            y_3 = y_3.append(df.iloc[12+48*seed+drop_r*16:16+48*seed+drop_r*16][s].reset_index(drop=True))
    
        
    # print("\nFAIRDROP 0.99 \n")

    # print(x_3)
    # print(y_3)

        c1 = "#CC0000"
        #c2 = "#FF0000"
        #c3 = "#FF6666"
        c2 = c1
        c3 = c1
    
    # x = x.append(pd.Series(x[0]))
    # y = y.append(pd.Series(y[0]))

    # x_1 = x_1.append(pd.Series(x_1[0]))
    # y_1 = y_1.append(pd.Series(y_1[0]))

    # x_2 = x_2.append(pd.Series(x_2[0]))
    # y_2 = y_2.append(pd.Series(y_2[0]))

    # x_3 = x_3.append(pd.Series(x_3[0]))
    # y_3 = y_3.append(pd.Series(y_3[0]))

        plt.scatter(x, y, c='blue', label="NIFTY", alpha=0.3)
        plt.scatter(x_1, y_1, c=c1, label="NIFTY + FAIRDROP" , alpha=0.3)
        plt.scatter(x_2, y_2, c=c2, label="NIFTY + FAIRDROP" , alpha=0.3)
        plt.scatter(x_3, y_3, c=c3, label="NIFTY + FAIRDROP" , alpha=0.3)

    # plt.fill(x,y, color='blue')

    # plt.fill(x_1, y_1, color=c1, alpha=0.5)

    # plt.fill(x_2, y_2, color=c2, alpha=0.4)

    # plt.fill(x_3, y_3, color=c3, alpha=0.3)


    
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)

        if drop_r == 0:
            dr = 0.01
        elif drop_r == 1:
            dr = 0.1
        elif drop_r == 2:
            dr = 0.2

        #t = title.replace('-', '').replace('longrun', '') + " - " + str(dr)
        #t = t.strip()
        t = title.replace('-', '').replace('longrun', '')
        t = t.strip()

    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.title(t)
    plt.xlabel("AUROC")
    plt.ylabel(s)
    plt.legend(newHandles, newLabels)
    plt.rcParams["figure.figsize"] = (10,6)

    plt.savefig("grafici_paper/" + t + "-" + s +  ".png")
    plt.close()


for i in [" credit-longrun", " german-longrun", " pokec_n-longrun", " bail-longrun"]:
    f = "output_longrun/output" + i + ".txt"
    with open(f, 'r') as file:
        lines = file.readlines()
        file.close()

    with open(f, 'r') as file, open("output_longrun/output" + i + "processed.txt", 'w') as file1:
        c = 0
        for number, line in enumerate(lines):
            if 'edges' not in line and 'pytorch' not in line and 'pokec' not in line and 'FAIRDROP' not in line:
                if 'AUCROC' in line:
                    A = float(re.findall("(?<=[AZaz])?(?!\d*=)[0-9.+-]+", line)[0])
                    file1.write('{:.4f}\t'.format(A))
                elif 'Parity' in line:
                    p = float(re.findall("(?<=[AZaz])?(?!\d*=)[0-9.+-]+", line)[0])
                    file1.write(f'{p}\t')
                    e = float(re.findall("(?<=[AZaz])?(?!\d*=)[0-9.+-]+", line)[1])
                    file1.write(f'{e}\t')
                elif 'F1' in line:
                    f1 = float(re.findall("(?<=[AZaz])?(?!\d*=)[0-9.+-]+", line)[1])
                    file1.write(f'{f1}\t')
                elif 'CounterFactual' in line:
                    cf = float(re.findall("(?<=[AZaz])?(?!\d*=)[0-9.+-]+", line)[0])
                    file1.write(f'{cf}\t')
                elif 'Robustness' in line: 
                    rob = float(re.findall("(?<=[AZaz])?(?!\d*=)[0-9.+-]+", line)[0])
                    file1.write(f'{rob}\n')
            else: c=c+1
        
            
        file.close()
        file1.close()

    df0 = pd.read_csv("output_longrun/output" + i + "processed.txt", sep='\t', lineterminator='\n', names=["AUROC", "Parity", "Equality", "F1", "Counterfactual", "Robustness"], index_col=False)
    
        
    makecloud(i, df0, True)
    makecloud(i, df0, False)

        #makecloud(i, df0, drop_r, False)


