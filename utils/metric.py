from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def create_matrix(y_trues, y_preds, out_dir, epoch):

    matrix = confusion_matrix(y_trues, y_preds, labels=[i for i in range(5)])
    df_mat = pd.DataFrame(matrix/np.sum(matrix))
    plt.figure(figsize = (12, 7))
    sns.heatmap(df_mat, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig(os.path.join(out_dir, f'val_{epoch}_output.png'))
    
def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
    of adoption rating."""
    w = np.zeros((N,N))
    O = confusion_matrix(actuals, preds, labels=[i for i in range(5)])
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)
    
    act_hist=np.zeros([N])
    for item in actuals: 
        act_hist[item]+=1
    
    pred_hist=np.zeros([N])
    for item in preds: 
        pred_hist[item]+=1
                         
    E = np.outer(act_hist, pred_hist)
    E = E/E.sum()
    O = O/O.sum()
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
    return (1 - (num/den))

def print_report(y_true, y_pred, target_names=[i for i in range(5)]):
    print(classification_report(y_true, y_pred, labels=[i for i in range(5)], zero_division=0))
    print("=" * 25 + 'Kappa Score' + '=' * 25)
    print(quadratic_kappa(y_true, y_pred))
