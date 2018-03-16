from sklearn.metrics import roc_auc_score
import cmath

def Evaluate_Fun(pred, y_test, x_test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    num = len(x_test)
    for i in range(num):
        if pred[i]==0 and y_test[i]==0:
            TP += 1
        elif pred[i]==1 and y_test[i]==1:
            TN += 1
        elif pred[i]==0 and y_test[i]==1:
            FP += 1
        elif pred[i]==1 and y_test[i]==0:
            FN += 1

    print('TP=',TP,',TN=', TN, ',FP=',FP,',FN=',FN)
    Accuracy = (TP + TN)/(TP + FP + TN + FN)
    MCC = (TP*TN -FP*FN)/(cmath.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    AUC = roc_auc_score(y_test, pred)
    SN = TP/(TP+FN)
    SP = TN/(TN+FP)
    Precision = TP/(TP+FP)
    print('ACC:',Accuracy)
    print('MCC:',MCC)
    print('AUC:',AUC)
    print('SN:',SN)
    print('SP:',SP)
    print('Precison:',Precision)
