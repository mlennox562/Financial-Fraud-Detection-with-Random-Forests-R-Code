library(knitr)
library(caret)
library(plyr)
library(dplyr)
library(xgboost)
library(ranger)
library(nnet)
library(Metrics)
library(ggplot2)
library(doMC)

registerDoMC(cores = 4)

knitr::opts_chunk$set(echo = TRUE, message=FALSE, warning=FALSE)

ROOT.DIR <- "C:/Users/Mark/Documents" #getwd()

DATA.DIR <- paste(ROOT.DIR,"Credit Card Data",sep="/")

test.raw.class <- read.csv("C:/Users/Mark/Documents/creditcard/test_data_Class.csv",stringsAsFactors = FALSE)

train.raw <- read.csv("C:/Users/Mark/Documents/creditcard/balancedData_SMOTE.csv",stringsAsFactors = FALSE)

#train.raw$Class <- as.factor(train.raw$Class)

train.raw$Class <- as.factor(ifelse(train.raw$Class == 0,'Good', 'Bad'))

test.raw <- read.csv("C:/Users/Mark/Documents/creditcard/test_NoClass.csv",stringsAsFactors = FALSE)

train.raw <- train.raw[,-1]

test.raw <- test.raw[,-1]

CONFIRMED_ATTR <- colnames(train.raw)

CONFIRMED_ATTR <- CONFIRMED_ATTR[2:31]

REJECTED_ATTR <-  setdiff(colnames(train.raw),CONFIRMED_ATTR)

PREDICTOR_ATTR <- c(CONFIRMED_ATTR,REJECTED_ATTR)

# Determine data types in the data set
data_types <- sapply(PREDICTOR_ATTR,function(x){class(train.raw[[x]])})
unique_data_types <- unique(data_types)

# Separate attributes by data type
DATA_ATTR_TYPES <- lapply(unique_data_types,function(x){ names(data_types[data_types == x])})
names(DATA_ATTR_TYPES) <- unique_data_types

# create folds for training
set.seed(13)
data_folds <- createFolds(train.raw$Class, k=5)

# Feature Set 1 - Boruta Confirmed and tentative Attributes
prepL0FeatureSet1 <- function(df) {
  id <- df$id
  if (class(df$Class) != "NULL") {
    y <- df$Class
  } else {
    y <- NULL
  }
  
  predictor_vars <- c(CONFIRMED_ATTR)
  
  predictors <- df[predictor_vars]
  
  # for numeric set missing values to -1 for purposes
  num_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$numeric)
  for (x in num_attr){
    predictors[[x]][is.na(predictors[[x]])] <- -1
  }
  
  # for character  atributes set missing value
  char_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$character)
  for (x in char_attr){
    predictors[[x]][is.na(predictors[[x]])] <- "*MISSING*"
    predictors[[x]] <- factor(predictors[[x]])
  }
  
  return(list(id=id,y=y,predictors=predictors))
}

L0FeatureSet1 <- list(train=prepL0FeatureSet1(train.raw),
                      test=prepL0FeatureSet1(test.raw))

# Feature Set 2 (xgboost) - Boruta Confirmed Attributes
prepL0FeatureSet2 <- function(df) {
  id <- df$id
  if (class(df$Class) != "NULL") {
    y <- df$Class
  } else {
    y <- NULL
  }
  
  
  predictor_vars <- c(CONFIRMED_ATTR)
  
  predictors <- df[predictor_vars]
  
  # for numeric set missing values to -1 for purposes
  num_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$numeric)
  for (x in num_attr){
    predictors[[x]][is.na(predictors[[x]])] <- -1
  }
  
  # for character  atributes set missing value
  char_attr <- intersect(predictor_vars,DATA_ATTR_TYPES$character)
  for (x in char_attr){
    predictors[[x]][is.na(predictors[[x]])] <- "*MISSING*"
    predictors[[x]] <- as.numeric(factor(predictors[[x]]))
  }
  
  return(list(id=id,y=y,predictors=as.matrix(predictors)))
}

L0FeatureSet2 <- list(train=prepL0FeatureSet2(train.raw),
                      test=prepL0FeatureSet2(test.raw))

#train model on one data fold
trainOneFold <- function(this_fold,feature_set) {
  # get fold specific cv data
  cv.data <- list()
  cv.data$predictors <- feature_set$train$predictors[this_fold,]
  cv.data$id <- feature_set$train$id[this_fold]
  cv.data$y <- feature_set$train$y[this_fold]
  
  # get training data for specific fold
  train.data <- list()
  train.data$predictors <- feature_set$train$predictors[-this_fold,]
  train.data$y <- feature_set$train$y[-this_fold]
  
  
  set.seed(825)
  fitted_mdl <- do.call(train,
                        c(list(x=train.data$predictors,y=train.data$y),
                          CARET.TRAIN.PARMS,
                          MODEL.SPECIFIC.PARMS,
                          CARET.TRAIN.OTHER.PARMS))
  
  yhat <- predict(fitted_mdl,newdata = cv.data$predictors,type = "prob")
  
  yhat <- as.factor(ifelse(yhat$Bad > 0.5,'Bad','Good'))
  
  precision <- posPredValue(yhat, cv.data$y, positive = "Bad")
  
  recall <- sensitivity(yhat, cv.data$y, positive = "Good")
  
  score <- (2 * precision * recall) / (precision + recall)
  
  ans <- list(fitted_mdl=fitted_mdl,
              score=score,
              predictions=data.frame(id=cv.data$id,yhat=yhat,y=cv.data$y))
  
  return(ans)
  
}

# make prediction from a model fitted to one fold
makeOneFoldTestPrediction <- function(this_fold,feature_set) {
  
  fitted_mdl <- this_fold$fitted_mdl
  
  yhat <- predict(fitted_mdl,newdata = feature_set$test$predictors,type = "prob")
  
  yhat <- as.factor(ifelse(yhat$Bad > 0.5,'Bad','Good'))
  
  return(yhat)
}

'============================================================================================================='

# set caret training parameters
CARET.TRAIN.PARMS <- list(method="gbm")   

CARET.TUNE.GRid <-  expand.grid(n.trees=1000, 
                                interaction.depth=20, 
                                shrinkage=0.1,
                                n.minobsinnode=10)

MODEL.SPECIFIC.PARMS <- list(verbose=0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRid,metric='f1')

# generate features for Level 1
gbm_set <- llply(data_folds,trainOneFold,L0FeatureSet1)

# final model fit
gbm_mdl <- do.call(train,
                   c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- as.factor(do.call(c,lapply(gbm_set,function(x){x$predictions$y})))

cv_y <- mapvalues(cv_y, from = c("1", "2"), to = c("0", "1"))

cv_yhat <- as.factor(do.call(c,lapply(gbm_set,function(x){x$predictions$yhat})))

cv_yhat <- mapvalues(cv_yhat, from = c("1", "2"), to = c("0", "1"))

precision <- posPredValue(cv_yhat, cv_y, positive = "1")

recall <- sensitivity(cv_yhat, cv_y, positive = "1")

score <- (2 * precision * recall) / (precision + recall)

score

cat("Average CV f1:",mean(do.call(c,lapply(gbm_set,function(x){x$score}))))

# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_gbm_yhat <- predict(gbm_mdl,newdata = L0FeatureSet1$test$predictors,type = "prob")

test_gbm_yhat <- as.factor(ifelse(test_gbm_yhat$Bad > 0.5,'1','0'))

precision <- posPredValue(test_gbm_yhat, as.factor(test.raw.class$Class), positive = "1")

recall <- sensitivity(test_gbm_yhat, as.factor(test.raw.class$Class), positive = "1")

score <- (2 * precision * recall) / (precision + recall)

cat("GBM Test F1 score:",score,"\n")

cat("GBM Train Confusion Matrix:")

confusionMatrix(cv_yhat, cv_y, mode = "prec_recall",positive = "1")

cat("GBM Test Confusion Matrix:")

confusionMatrix(test_gbm_yhat,  as.factor(test.raw.class$Class), mode = "prec_recall",positive = "1")

gbm_submission <- cbind(id=L0FeatureSet1$test$id,Class=test_gbm_yhat)
 
gbm_submission <- mapvalues(gbm_submission, from = c(1, 2), to = c(0, 1))

#write.csv(gbm_submission,file="C:/Users/Mark/Documents/creditcard/gbm_submission.csv",row.names=FALSE)

'============================================================================================================='

# set caret training parameters
CARET.TRAIN.PARMS <- list(method="xgbTree")   

CARET.TUNE.GRID <-  expand.grid(nrounds=1000, 
                                max_depth=6, 
                                eta=0.3, 
                                gamma=1.34, 
                                colsample_bytree=0.678, 
                                min_child_weight=6.22)

MODEL.SPECIFIC.PARMS <- list(verbose=0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=TRUE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="f1")

# generate Level 1 features
xgb_set <- llply(data_folds,trainOneFold,L0FeatureSet2)

# final model fit
xgb_mdl <- do.call(train,
                   c(list(x=L0FeatureSet2$train$predictors,y=L0FeatureSet2$train$y),
                     CARET.TRAIN.PARMS,
                     MODEL.SPECIFIC.PARMS,
                     CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- as.factor(do.call(c,lapply(xgb_set,function(x){x$predictions$y})))

#cv_y <- mapvalues(cv_y, from = c("1", "2"), to = c("0", "1"))

cv_yhat <- as.factor(do.call(c,lapply(xgb_set,function(x){x$predictions$yhat})))

#cv_yhat <- mapvalues(cv_yhat, from = c("1", "2"), to = c("0", "1"))

precision <- posPredValue(cv_yhat, cv_y, positive = "2")

recall <- sensitivity(cv_yhat, cv_y, positive = "2")

score <- (2 * precision * recall) / (precision + recall)

score

cat("Average CV f1:",mean(do.call(c,lapply(xgb_set,function(x){x$score}))))

# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_xgb_yhat <- predict(xgb_mdl,newdata = L0FeatureSet2$test$predictors,type = "prob")

test_xgb_yhat <- as.factor(ifelse(test_xgb_yhat$Bad > 0.5,'1','0'))

precision <- posPredValue(test_xgb_yhat, as.factor(test.raw.class$Class), positive = "1")

recall <- sensitivity(test_xgb_yhat, as.factor(test.raw.class$Class), positive = "1")

score <- (2 * precision * recall) / (precision + recall)

cat("XGB Test F1 score:",score,"\n")

cat("XGB Train Confusion Matrix:")

confusionMatrix(cv_yhat, cv_y, mode = "prec_recall",positive = "1")

cat("XGB Test Confusion Matrix:")

confusionMatrix(test_xgb_yhat,  as.factor(test.raw.class$Class), mode = "prec_recall",positive = "1")

#xgb_submission <- cbind(id=L0FeatureSet2$test$id,Class=test_xgb_yhat)

#xgb_submission <- mapvalues(xgb_submission, from = c(1, 2), to = c(0, 1))

#write.csv(xgb_submission,file="C:/Users/Mark/Documents/Credit Card Data/xgb_sumbission.csv",row.names=FALSE)

'===================================================================================================================='

# set caret training parameters
CARET.TRAIN.PARMS <- list(method="ranger")   

CARET.TUNE.GRID <-  expand.grid(mtry=2*as.integer(sqrt(ncol(L0FeatureSet1$train$predictors))))

MODEL.SPECIFIC.PARMS <- list(verbose=0,num.trees=500) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="none",
                                 verboseIter=FALSE,
                                 classProbs=TRUE)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                tuneGrid=CARET.TUNE.GRID,
                                metric="f1")

# generate Level 1 features
rngr_set <- llply(data_folds,trainOneFold,L0FeatureSet1)

# final model fit
rngr_mdl <- do.call(train,
                    c(list(x=L0FeatureSet1$train$predictors,y=L0FeatureSet1$train$y),
                      CARET.TRAIN.PARMS,
                      MODEL.SPECIFIC.PARMS,
                      CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- as.factor(do.call(c,lapply(rngr_set,function(x){x$predictions$y})))

cv_y <- mapvalues(cv_y, from = c("1", "2"), to = c("0", "1"))

cv_yhat <- as.factor(do.call(c,lapply(rngr_set,function(x){x$predictions$yhat})))

cv_yhat <- mapvalues(cv_yhat, from = c("1", "2"), to = c("0", "1"))

precision <- posPredValue(cv_yhat, cv_y, positive = "1")

recall <- sensitivity(cv_yhat, cv_y, positive = "1")

score <- (2 * precision * recall) / (precision + recall)

score

cat("Average CV f1:",mean(do.call(c,lapply(rngr_set,function(x){x$score}))))

# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_rngr_yhat <- predict(rngr_mdl,newdata = L0FeatureSet1$test$predictors,type = "prob")

test_rngr_yhat <- as.factor(ifelse(test_rngr_yhat$Bad > 0.5,'1','0'))

precision <- posPredValue(test_rngr_yhat, as.factor(test.raw.class$Class), positive = "1")

recall <- sensitivity(test_rngr_yhat, as.factor(test.raw.class$Class), positive = "1")

score <- (2 * precision * recall) / (precision + recall)

cat("RNGR Test F1 score:",score,"\n")

cat("RNGR Train Confusion Matrix:")

confusionMatrix(cv_yhat, cv_y, mode = "prec_recall",positive = "1")

cat("RNGR Test Confusion Matrix:")

confusionMatrix(test_rngr_yhat,  as.factor(test.raw.class$Class), mode = "prec_recall",positive = "1")

#rngr_submission <- cbind(id=L0FeatureSet1$test$id,Class=test_rngr_yhat)

#rngr_submission <- mapvalues(rngr_submission, from = c(1, 2), to = c(0, 1))

#write.csv(rngr_submission,file="C:/Users/Mark/Documents/Credit Card Data/rngr_sumbission.csv",row.names=FALSE)

'================================================================================================================='

#gbm_yhat <- as.factor(do.call(c,lapply(gbm_set,function(x){x$predictions$yhat})))
gbm_yhat <- mapvalues(as.factor(do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))), from = c("1", "2"), to = c("0", "1"))
#xgb_yhat <- as.factor(do.call(c,lapply(xgb_set,function(x){x$predictions$yhat})))
xgb_yhat <- mapvalues(as.factor(do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))), from = c("1", "2"), to = c("0", "1"))
#rngr_yhat <- as.factor(do.call(c,lapply(rngr_set,function(x){x$predictions$yhat})))
rngr_yhat <- mapvalues(as.factor(do.call(c,lapply(rngr_set,function(x){x$predictions$yhat}))), from = c("1", "2"), to = c("0", "1"))

# gbm_yhat <- do.call(c,lapply(gbm_set,function(x){x$predictions$yhat}))
# xgb_yhat <- do.call(c,lapply(xgb_set,function(x){x$predictions$yhat}))
# rngr_yhat <- do.call(c,lapply(rngr_set,function(x){x$predictions$yhat}))

# create Feature Set
L1FeatureSet <- list()

L1FeatureSet$train$id <- do.call(c,lapply(gbm_set,function(x){x$predictions$id}))
#L1FeatureSet$train$y <- as.factor(do.call(c,lapply(gbm_set,function(x){x$predictions$y})))
L1FeatureSet$train$y <- mapvalues(as.factor(do.call(c,lapply(gbm_set,function(x){x$predictions$y}))), from = c("1", "2"), to = c("Good", "Bad"))

predictors <- data.frame(gbm_yhat,xgb_yhat,rngr_yhat)
predictors_rank <- t(apply(predictors,1,rank))
colnames(predictors_rank) <- paste0("rank_",names(predictors))
L1FeatureSet$train$predictors <- predictors #cbind(predictors,predictors_rank)

L1FeatureSet$test$id <- gbm_submission[,"id"]
L1FeatureSet$test$predictors <- data.frame(gbm_yhat=test_gbm_yhat,
                                           xgb_yhat=test_xgb_yhat,
                                           rngr_yhat=test_rngr_yhat)

'==================================================================================================================='

# set caret training parameters
CARET.TRAIN.PARMS <- list(method="rf") 

CARET.TUNE.GRID <-  NULL  # NULL provides model specific default tuning parameters

#model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method="repeatedcv",
                                 number=5,
                                 repeats=1,
                                 verboseIter=FALSE,
                                 summaryFunction = twoClassSummary,
                                 classProbs = TRUE)

# CARET.TRAIN.CTRL <- trainControl(method = 'cv',
#                                  number=5,
#                                  repeats=1,
#                                  classProbs = TRUE,
#                                  verboseIter = TRUE,
#                                  summaryFunction = twoClassSummary)

CARET.TRAIN.OTHER.PARMS <- list(trControl=CARET.TRAIN.CTRL,
                                maximize=FALSE,
                                tuneGrid=CARET.TUNE.GRID,
                                tuneLength=7,
                                metric="ROC")

MODEL.SPECIFIC.PARMS <- list(verbose=FALSE,linout=TRUE,trace=FALSE) #NULL # Other model specific parameters

# train the model
set.seed(825)
l1_rf_mdl <- do.call(train,c(list(x=L1FeatureSet$train$predictors,y=L1FeatureSet$train$y),
                             CARET.TRAIN.PARMS,
                             MODEL.SPECIFIC.PARMS,
                             CARET.TRAIN.OTHER.PARMS))

l1_rf_mdl

precision <- posPredValue(as.factor(ifelse(predict(l1_rf_mdl,newdata = L1FeatureSet$train$predictors,type = "prob")$Bad > 0.5,'Bad','Good')), L1FeatureSet$train$y, positive = "Bad")

recall <- sensitivity(as.factor(ifelse(predict(l1_rf_mdl,newdata = L1FeatureSet$train$predictors,type = "prob")$Bad > 0.5,'Bad','Good')), L1FeatureSet$train$y, positive = "Bad")

score <- (2 * precision * recall) / (precision + recall)

cat("Train F1 score:",score,"\n")

precision <- posPredValue(as.factor(ifelse(predict(l1_rf_mdl,newdata = L1FeatureSet$test$predictors,type = "prob")$Bad > 0.5,'1','0')), as.factor(test.raw.class$Class),positive = "1")

recall <- sensitivity(as.factor(ifelse(predict(l1_rf_mdl,newdata = L1FeatureSet$test$predictors,type = "prob")$Bad > 0.5,'1','0')), as.factor(test.raw.class$Class),positive = "1")

score <- (2 * precision * recall) / (precision + recall)

cat("Test F1 score:",score,"\n")

confusionMatrix(as.factor(ifelse(predict(l1_rf_mdl,newdata = L1FeatureSet$train$predictors,type = "prob")$Bad > 0.5,'Bad','Good')),  L1FeatureSet$train$y, mode = "prec_recall",positive = "Bad")

confusionMatrix(as.factor(ifelse(predict(l1_rf_mdl,newdata = L1FeatureSet$test$predictors,type = "prob")$Bad > 0.5,'1','0')), as.factor(test.raw.class$Class), mode = "prec_recall",positive = "1")

test_l1_rf_yhat <- as.factor(ifelse(predict(l1_rf_mdl,newdata = L1FeatureSet$test$predictors,type = "prob")$Bad > 0.5,'1','0'))

l1_rf_submission <- cbind(id=L1FeatureSet$test$id,Class=test_l1_rf_yhat)

l1_rf_submission <- as.data.frame(l1_rf_submission)

l1_rf_submission$Class <- mapvalues(l1_rf_submission$Class, from = c(1, 2), to = c(0, 1))

colnames(l1_rf_submission) <- c("id","Class")

#write.csv(l1_rf_submission,file="C:/Users/Mark/Documents/Credit Card Data/l1_rf_submission.csv",row.names=FALSE)
