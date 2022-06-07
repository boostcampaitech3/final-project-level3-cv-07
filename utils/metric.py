from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def create_matrix(y_trues, y_preds, out_dir, epoch, num_classes):
    out_dir = os.path.join(out_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    matrix = confusion_matrix(y_trues, y_preds, labels=[i for i in range(num_classes)])
    df_mat = pd.DataFrame(matrix)
    plt.figure(figsize = (12, 7))
    sns.heatmap(df_mat, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.savefig(os.path.join(out_dir, f'val_{epoch}_output.png'))
    
def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
    of adoption rating.
    
    """
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

def print_report(y_true, y_pred, num_classes):
    new_report = {}
    print(classification_report(y_true, y_pred, labels=[i for i in range(num_classes)], zero_division=0))
    report = classification_report(y_true, y_pred, labels=[i for i in range(num_classes)], output_dict=True)
    for label_num in [str(i) for i in range(num_classes)]:
        for metric in report[label_num].keys():
            new_name = str(label_num) + '_' + metric
            new_report[new_name] = report[str(label_num)][metric]
        new_report['macro_precision'] = report['macro avg']['precision']
    new_report['macro_recall'] = report['macro avg']['recall']
    new_report['macro_f1_score'] = report['macro avg']['f1-score']
    print("=" * 25 + 'Kappa Score' + '=' * 25)
    kappa = quadratic_kappa(y_true, y_pred, N=num_classes)
    print(kappa)
    new_report['kappa'] = kappa
    return new_report

