import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import randn

def plot_f1(logs):
    markers = ("o", "x", "s", "^")
    colors = ('dodgerblue','mediumseagreen', 'hotpink', '#fba84a')
    
    def plot_train():
        plt.rcParams["figure.figsize"] = (16,4)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.color'] = 'r'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        plt.suptitle('f1 - train', fontsize=20)

        for i, k in enumerate(logs):
            # epoch/loss/acc/f1/tp/tn/fp/fn
            val = pd.read_csv(f"{logs[k]}/val.tsv", delimiter='\t', header=None, skiprows=1)
            train = pd.read_csv(f"{logs[k]}/train.tsv", delimiter='\t', header=None, skiprows=1)
            
            df = train
            
            loss = df[1].to_numpy()
            f1 = df[3].to_numpy()
            tp = df[4].to_numpy()
            tn = df[5].to_numpy()
            fp = df[6].to_numpy()
            fn = df[7].to_numpy()
            pre = tp / (tp+fp)
            rec = tp / (tp+tn)
            
            plt.plot(f1, color=colors[i], marker=markers[i])
        plt.legend(logs.keys(), fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('f1', fontsize=15)
        plt.show()
    def plot_val():
        plt.rcParams["figure.figsize"] = (16,4)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.color'] = 'r'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        plt.suptitle('f1 - val', fontsize=20)

        for i, k in enumerate(logs):
            # epoch/loss/acc/f1/tp/tn/fp/fn
            val = pd.read_csv(f"{logs[k]}/val.tsv", delimiter='\t', header=None, skiprows=1)
            train = pd.read_csv(f"{logs[k]}/train.tsv", delimiter='\t', header=None, skiprows=1)
            
            df = val
            
            loss = df[1].to_numpy()
            f1 = df[3].to_numpy()
            tp = df[4].to_numpy()
            tn = df[5].to_numpy()
            fp = df[6].to_numpy()
            fn = df[7].to_numpy()
            pre = tp / (tp+fp)
            rec = tp / (tp+tn)
            
            plt.plot(f1, color=colors[i], marker=markers[i])
        plt.legend(logs.keys(), fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('f1', fontsize=15)
        plt.show()
    plot_train()
    plot_val()
    
def plot_loss(logs):
    markers = ("o", "x", "s", "^")
    colors = ('dodgerblue','mediumseagreen', 'hotpink', '#fba84a')
    
    def plot_train():
        plt.rcParams["figure.figsize"] = (16,4)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.color'] = 'r'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        plt.suptitle('loss - train', fontsize=20)

        for i, k in enumerate(logs):
            # epoch/loss/acc/f1/tp/tn/fp/fn
            val = pd.read_csv(f"{logs[k]}/val.tsv", delimiter='\t', header=None, skiprows=1)
            train = pd.read_csv(f"{logs[k]}/train.tsv", delimiter='\t', header=None, skiprows=1)
            
            df = train
            
            loss = df[1].to_numpy()
            f1 = df[3].to_numpy()
            tp = df[4].to_numpy()
            tn = df[5].to_numpy()
            fp = df[6].to_numpy()
            fn = df[7].to_numpy()
            pre = tp / (tp+fp)
            rec = tp / (tp+tn)
            
            plt.plot(loss, color=colors[i], marker=markers[i])
        plt.legend(logs.keys(), fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('loss', fontsize=15)
        plt.show()
    def plot_val():
        plt.rcParams["figure.figsize"] = (16,4)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.color'] = 'r'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        plt.suptitle('loss - val', fontsize=20)
    
        for i, k in enumerate(logs):
            # epoch/loss/acc/f1/tp/tn/fp/fn
            val = pd.read_csv(f"{logs[k]}/val.tsv", delimiter='\t', header=None, skiprows=1)
            train = pd.read_csv(f"{logs[k]}/train.tsv", delimiter='\t', header=None, skiprows=1)
            
            df = val
            
            loss = df[1].to_numpy()
            f1 = df[3].to_numpy()
            tp = df[4].to_numpy()
            tn = df[5].to_numpy()
            fp = df[6].to_numpy()
            fn = df[7].to_numpy()
            pre = tp / (tp+fp)
            rec = tp / (tp+tn)
            
            plt.plot(loss, color=colors[i], marker=markers[i])
        plt.legend(logs.keys(), fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('loss', fontsize=15)
        plt.show()
    plot_train()
    plot_val()
    
    
def plot_custom(logs):
    markers = ("o", "x", "s", "^")
    colors = ('dodgerblue','mediumseagreen', 'hotpink', '#fba84a')
    
    def plot_train():
        plt.rcParams["figure.figsize"] = (16,4)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.color'] = 'r'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        plt.suptitle('custom - train', fontsize=20)

        for i, k in enumerate(logs):
            # epoch/loss/acc/f1/tp/tn/fp/fn
            val = pd.read_csv(f"{logs[k]}/val.tsv", delimiter='\t', header=None, skiprows=1)
            train = pd.read_csv(f"{logs[k]}/train.tsv", delimiter='\t', header=None, skiprows=1)
            
            df = train
            
            loss = df[2].to_numpy()
            f1 = df[3].to_numpy()
            tp = df[4].to_numpy()
            tn = df[5].to_numpy()
            fp = df[6].to_numpy()
            fn = df[7].to_numpy()
            pre = tp / (tp+fp)
            rec = tp / (tp+tn)
            
            plt.plot(pre/rec, color=colors[i], marker=markers[i])
        plt.legend(logs.keys(), fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('custom', fontsize=15)
        plt.show()
    def plot_val():
        plt.rcParams["figure.figsize"] = (16,4)
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.color'] = 'r'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        plt.suptitle('custom - val', fontsize=20)
    
        for i, k in enumerate(logs):
            # epoch/loss/acc/f1/tp/tn/fp/fn
            val = pd.read_csv(f"{logs[k]}/val.tsv", delimiter='\t', header=None, skiprows=1)
            train = pd.read_csv(f"{logs[k]}/train.tsv", delimiter='\t', header=None, skiprows=1)
            
            df = val
            
            loss = df[2].to_numpy()
            f1 = df[3].to_numpy()
            tp = df[4].to_numpy()
            tn = df[5].to_numpy()
            fp = df[6].to_numpy()
            fn = df[7].to_numpy()
            pre = tp / (tp+fp)
            rec = tp / (tp+tn)
            
            plt.plot(pre/rec, color=colors[i], marker=markers[i])
        plt.legend(logs.keys(), fontsize=15)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('custom', fontsize=15)
        plt.show()
    plot_train()
    plot_val()