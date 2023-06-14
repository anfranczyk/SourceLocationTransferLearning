function mat = myConf2D(TN, FP, FN, TP)
% kolejnoœæ argumentów jak w matlabie
sensitivity = TP/(TP+FN)
specificity = TN / (TN+FP)
acuraccy = (TP+TN)/(TN+FP+FN+TP)
precission = TP/(TP+FP)
negPred = TN/(TN+FN)

mat = [ TP FN sensitivity; FP TN,  specificity';  precission  negPred acuraccy ];
end