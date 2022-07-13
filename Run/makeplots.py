from cProfile import label
from numpy import float64
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.ticker import PercentFormatter
from scipy.spatial import ConvexHull



def makecloud(title, df, parity = True):
    
    if parity: s = "Parity"
    else: s = "Equality"


    x = df.iloc[0::36]['AUROC'].reset_index(drop=True)
    y = df.iloc[0::36][s].reset_index(drop=True)

    #print("\nNIFTY\n")
    #print(x)
    #print(y)

    x_1 = df.iloc[2::36]['AUROC'].reset_index(drop=True)
    y_1 = df.iloc[2::36][s].reset_index(drop=True)
    
    #print("\n FAIRDROP 0.75 \n")

    #print(x_1)
    #print(y_1)
    
    x_2 = df.iloc[6::36]['AUROC'].reset_index(drop=True)
    y_2 = df.iloc[6::36][s].reset_index(drop=True)
    
    #print("\n FAIRDROP 0.90 \n")

    #print(x_2)
    #print(y_2)

    x_3 = df.iloc[10::36]['AUROC'].reset_index(drop=True)
    y_3 = df.iloc[10::36][s].reset_index(drop=True)

    for drop_r in [1,2]:
    #print("\n ~~~~~~~~~~~~~~~~", drop_r, "~~~~~~~~~~~~~~~~~~ì\n")

        x = x.append(df.iloc[0+drop_r*12::36]['AUROC'].reset_index(drop=True))
        y = y.append(df.iloc[0+drop_r*12::36][s].reset_index(drop=True))

        print("\nNIFTY\n")
        print(x)
        print(y)

        x_1 = x_1.append(df.iloc[2+drop_r*12::36]['AUROC'].reset_index(drop=True))
        y_1 = y_1.append(df.iloc[2+drop_r*12::36][s].reset_index(drop=True))
    
    #print("\n FAIRDROP 0.75 \n")

    #print(x_1)
    #print(y_1)
    
        x_2 = x_2.append(df.iloc[6+drop_r*12::36]['AUROC'].reset_index(drop=True))
        y_2 = y_2.append(df.iloc[6+drop_r*12::36][s].reset_index(drop=True))
    
    #print("\n FAIRDROP 0.90 \n")

    #print(x_2)
    #print(y_2)

        x_3 = x_3.append(df.iloc[10+drop_r*12::36]['AUROC'].reset_index(drop=True))
        y_3 = y_3.append(df.iloc[10+drop_r*12::36][s].reset_index(drop=True))
    
    #print("\nFAIRDROP 0.99 \n")

    #print(x_3)
    #print(y_3)

        c1 = "#CC0000"
        c2 = "#FF0000"
        c3 = "#FF6666"

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

    plt.scatter(x, y, c='blue', label="NIFTY")
    plt.scatter(x_1, y_1, c=c1, label="NIFTY + FAIRDROP" )
    # plt.scatter(x_2, y_2, c=c2, label="NIFTY + FAIRDROP 0.49")
    # plt.scatter(x_3, y_3, c=c3, label="NIFTY + FAIRDROP 0.499")

    plt.scatter(x_2, y_2, c=c2 )
    plt.scatter(x_3, y_3, c=c3 )

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
        #t = "German"

    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.title(t)
    plt.xlabel("AUROC")
    plt.ylabel(s)
    plt.legend()
    plt.rcParams["figure.figsize"] = (10,6)

    plt.savefig("grafici_paper_final/" + t + "-" + s +  ".png")
    plt.close()


def makecloud2(title, df0, df1, parity = True):
    
    if parity: 
        s = 4
        m = "DP"
    else:
        s = 5
        m = "EO"

    #nifty
    x = [float64(xx) for xx in df0[3::7]]
    y = [float64(yy) for yy in df0[s::7]]

    #fairattr
    x1 = [float64(xx) for xx in df1[3::7]]
    y1 = [float64(yy) for yy in df1[s::7]]
    c1 = "#CC0000"
    c2 = "#FF0000"
    c3 = "#FF6666"
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

    plt.scatter(x, y, c='blue', label="NIFTY")
    plt.scatter(x1, y1, c='red', label="NIFTY + FAIRRF" )
    # plt.scatter(x_2, y_2, c=c2, label="NIFTY + FAIRDROP 0.49")
    # plt.scatter(x_3, y_3, c=c3, label="NIFTY + FAIRDROP 0.499")

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

        #t = title.replace('-', '').replace('longrun', '') + " - " + str(dr)
        #t = t.strip()

        t = title.replace('pulito', '').replace('longrun', '-' + m)
        t = t.strip()
        #t = "German"

    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.gca().xaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.title(t)
    plt.xlabel("AUC")
    plt.ylabel(s)
    plt.legend()
    plt.rcParams["figure.figsize"] = (10,6)

    plt.savefig("../grafici_paper_final/" + t + "-" + m +  "-fairrf.png")
    plt.close()




#for i in ["-bail-longrun", "-credit-longrun", "-german-longrun", "-pokec_n-longrun"]:
#for i in ["-pokec_n-longrun-0.47-0.49-0.499"]:
for i in ["germanpulito","bailpulito","creditpulito"]:
    f = i + ".txt"
    f1 = i + "-nifty.txt"
    with open(f, 'r') as file:
        df1 = list(filter(None, file.read().splitlines()))
        with open(f1, 'r') as file1:
            df0 = list(filter(None, file1.read().splitlines()))
            makecloud2(i, df0, df1,  False)
            