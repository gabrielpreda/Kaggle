library(randomForest)
library(caret)
library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)

#this Kernel uses and modifies code from https://www.r-bloggers.com/illustrated-guide-to-roc-and-auc/

#calculate ROC (https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
calculate_roc <- function(verset, cost_of_fp, cost_of_fn, n=100) {
  
  tp <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 1)
  }
  
  fp <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 0)
  }
  
  tn <- function(verset, threshold) {
    sum(verset$predicted < threshold & verset$Class == 0)
  }
  
  fn <- function(verset, threshold) {
    sum(verset$predicted < threshold & verset$Class == 1)
  }
  
  tpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 1) / sum(verset$Class == 1)
  }
  
  fpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 0) / sum(verset$Class == 0)
  }
  
  cost <- function(verset, threshold, cost_of_fp, cost_of_fn) {
    sum(verset$predicted >= threshold & verset$Class == 0) * cost_of_fp + 
      sum(verset$predicted < threshold & verset$Class == 1) * cost_of_fn
  }
  fpr <- function(verset, threshold) {
    sum(verset$predicted >= threshold & verset$Class == 0) / sum(verset$Class == 0)
  }
  
  threshold_round <- function(value, threshold)
  {
    return (as.integer(!(value < threshold)))
  }
  #calculate AUC (https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)
  auc_ <- function(verset, threshold) {
    auc(verset$Class, threshold_round(verset$predicted,threshold))
  }
  
  roc <- data.frame(threshold = seq(0,1,length.out=n), tpr=NA, fpr=NA)
  roc$tp <- sapply(roc$threshold, function(th) tp(verset, th))
  roc$fp <- sapply(roc$threshold, function(th) fp(verset, th))
  roc$tn <- sapply(roc$threshold, function(th) tn(verset, th))
  roc$fn <- sapply(roc$threshold, function(th) fn(verset, th))
  roc$tpr <- sapply(roc$threshold, function(th) tpr(verset, th))
  roc$fpr <- sapply(roc$threshold, function(th) fpr(verset, th))
  roc$cost <- sapply(roc$threshold, function(th) cost(verset, th, cost_of_fp, cost_of_fn))
  roc$auc <-  sapply(roc$threshold, function(th) auc_(verset, th))
  
  return(roc)
}

plot_roc <- function(roc, threshold, cost_of_fp, cost_of_fn) {
  library(gridExtra)
  
  norm_vec <- function(v) (v - min(v))/diff(range(v))
  
  idx_threshold = which.min(abs(roc$threshold-threshold))
  
  col_ramp <- colorRampPalette(c("green","orange","red","black"))(100)
  col_by_cost <- col_ramp[ceiling(norm_vec(roc$cost)*99)+1]
  p_roc <- ggplot(roc, aes(fpr,tpr)) + 
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=2, alpha=0.5) +
    labs(title = sprintf("ROC")) + xlab("FPR") + ylab("TPR") +
    geom_hline(yintercept=roc[idx_threshold,"tpr"], alpha=0.5, linetype="dashed") +
    geom_vline(xintercept=roc[idx_threshold,"fpr"], alpha=0.5, linetype="dashed")
  
  p_auc <- ggplot(roc, aes(threshold, auc)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=2, alpha=0.5) +
    labs(title = sprintf("AUC")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")
  
  p_cost <- ggplot(roc, aes(threshold, cost)) +
    geom_line(color=rgb(0,0,1,alpha=0.3)) +
    geom_point(color=col_by_cost, size=2, alpha=0.5) +
    labs(title = sprintf("cost function")) +
    geom_vline(xintercept=threshold, alpha=0.5, linetype="dashed")
  
  sub_title <- sprintf("threshold at %.2f - cost of FP = %d, cost of FN = %d", threshold, cost_of_fp, cost_of_fn)
  # 
  grid.arrange(p_roc, p_auc, p_cost, ncol=2,sub=textGrob(sub_title, gp=gpar(cex=1), just="bottom"))
}

raw.data <- read.csv("../input/creditcard.csv")
#raw.data <- read.csv("creditcard.csv")
nrows <- nrow(raw.data)
set.seed(314)
indexT <- sample(1:nrow(raw.data), 0.7 * nrows)

#separate train and validation set
trainset = raw.data[indexT,]
verset =   raw.data[-indexT,]

n <- names(trainset)
rf.form <- as.formula(paste("Class ~", paste(n[!n %in% "Class"], collapse = " + ")))

trainset.rf <- randomForest(rf.form,trainset,ntree=100,importance=T)

varImpPlot(trainset.rf, sort = T,main="Variable Importance",n.var=10)

library(pROC)

threshold = 0.1
verset$predicted <- predict(trainset.rf ,verset)
roc <- calculate_roc(verset, 1, 10, n = 100)
plot_roc(roc, threshold, 1, 10)
