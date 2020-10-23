import matplotlib.pyplot as plt
import itertools
import numpy as np
import seaborn as sns
from sklearn.metrics import auc
from numpy import set_printoptions

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Creates a plot for the specified confusion matrix object and calculates relevant accuracy measures.
    """

    # Add Normalization option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fp_label = 'false positive'
    fp = cm[0][1]
    fn_label = 'false negative'
    fn = cm[1][0]
    tp_label = 'true positive'
    tp = cm[1][1]
    tn_label = 'true negative'
    tn = cm[0][0]

    tpr_label = 'sensitivity'
    tpr = round(tp / (tp + fn), 3)
    tnr_label = 'specificity'
    tnr = round(tn / (tn + fp), 3)
    ppv_label = 'precision'
    ppv = round(tp / (tp + fp), 3)
    npv_label = 'npv'
    npv = round(tn / (tn + fn), 3)
    fpr_label = 'fpr'
    fpr = round(fp / (fp + tn), 3)
    fnr_label = 'fnr'
    fnr = round(fn / (tp + fn), 3)
    fdr_label = 'fdr'
    fdr = round(fp / (tp + fp), 3)

    acc_score = round((tp + tn) / (tp + fp + tn + fn), 3)

    print('\naccuracy:\t\t\t{}  \nprecision:\t\t\t{} \nrecall:\t\t\t\t{}'.format(acc_score, ppv, tpr))
    print('\nspecificity:\t\t\t{} \nnegative predictive value:\t{}'.format(tnr, npv))
    print('\nfalse positive rate:\t\t{}  \nfalse negative rate:\t\t{} \nfalse discovery rate:\t\t{}'.format(fpr, fnr,
                                                                                                            fdr))
def plot_roc_curve(fpr, tpr, title="Receiver operating characteristic (ROC) Curve"):
    """
    Creates a plot for the specified roc curve object.
    """

    # Visualization for ROC curve
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set(font_scale=1);
    print('AUC: {}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10,8))
    lw = 2
    _ = plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve');
    _ = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--');
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    _ = plt.show();

def plot_feature_importance_log(fit, features):
    """
    Creates a plot for the specified feature importance object.
    """

    set_printoptions(precision=3)

    # Summarize selected features
    scores = -np.log10(fit.pvalues)
    #scores /= scores.max()

    importances = np.array(scores)
    feature_list = features
    sorted_ID=np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

    for i,v in enumerate(reverse_importances):
        print('Feature: %10s\tScore:\t%.5f' % (reverse_features[i],v))

    # Plot feature importance
    #sorted_ID=np.array(np.argsort(scores)[::-1])
    sns.set(font_scale=1);
    _ = plt.figure(figsize=[10,10]);
    _ = plt.xticks(rotation='horizontal')
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.show();

    #_=plt.bar(X_indices - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)')
    #plt.show()
    # Summarize feature importance

def plot_feature_importance(fit, features):
    """
    Creates a plot for the specified feature importance object.
    """

    set_printoptions(precision=3)

    # Summarize selected features
    scores = -np.log10(fit.pvalues_)
    #scores /= scores.max()

    importances = np.array(scores)
    feature_list = features
    sorted_ID=np.array(np.argsort(scores))
    reverse_features = feature_list[sorted_ID][::-1]
    reverse_importances = importances[sorted_ID][::-1]

    for i,v in enumerate(reverse_importances):
        print('Feature: %10s\tScore:\t%.5f' % (reverse_features[i],v))

    # Plot feature importance
    #sorted_ID=np.array(np.argsort(scores)[::-1])
    sns.set(font_scale=1);
    _ = plt.figure(figsize=[10,10]);
    _ = plt.xticks(rotation='horizontal')
    _ = plt.barh(feature_list[sorted_ID], importances[sorted_ID], align='center');
    _ = plt.show();

    #_=plt.bar(X_indices - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)')
    #plt.show()
    # Summarize feature importance

    def main():
        from sklearn.metrics import confusion_matrix
        """
        main function - does all the work
        """
        # parse arguments
        cnf_matrix = confusion_matrix([0, 0, 1, 1], [0, 0, 1, 1])

        # generate plots
        plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True)

    if __name__ == "__main__":
        # call main
        main()