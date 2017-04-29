#Timing the entire procedure to compare with earlier classifiers.

start.time <- Sys.time()

#Adapted from the caret vignette
library("caret")
library("mlbench")
library("pROC")

library(doMC)
library(parallelMap)
library(parallel)

registerDoMC(cores = detectCores())

'================================================================================================================='

#The training and testing datasets are read into R and then prepared for the model building phase.

test.raw.class <- read.csv("C:/Users/Mark/Documents/creditcard/test_data_Class.csv",stringsAsFactors = FALSE)

train.raw <- read.csv("C:/Users/Mark/Documents/creditcard/train.raw.subset.csv",stringsAsFactors = FALSE)

train_id <- train.raw$id

train.raw$Class <- as.factor(train.raw$Class)

#train.raw$Class <- as.factor(ifelse(train.raw$Class==0, "Good", "Bad"))

test.raw <- read.csv("C:/Users/Mark/Documents/creditcard/test_NoClass.csv",stringsAsFactors = FALSE)

test_id <- test.raw$id

#train.raw <- train.raw[,-1]

test.raw <- test.raw[,-1]

train.raw <- train.raw[,-1]

test.raw <- test.raw[,-1]

test.raw$Time<- as.numeric(test.raw$Time)

train.raw$Time<- as.numeric(train.raw$Time)

#Test whether normalising the Amount and Time variables helps prediction accuracy. 

#train.raw$Time <- scale(train.raw$Time)

#test.raw$Time <- scale(test.raw$Time)

#train.raw$Amount <- scale(train.raw$Amount)

#test.raw$Amount <- scale(test.raw$Amount)

'================================================================================================================='

# my_control <- trainControl(
#   method="boot",
#   number=25,
#   savePredictions="final",
#   classProbs=TRUE,
#   index=createResample(train.raw$Class, 25),
#   summaryFunction=twoClassSummary,
#   allowParallel=TRUE
# )

#Cross-Validated training data 5 times. 

my_control <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction=twoClassSummary,
  verboseIter=TRUE,
  classProbs=TRUE,
  allowParallel=TRUE
)

#Functions for creating ensembles of caret models.

library("caretEnsemble")
model_list <- caretList(
  Class~., data=train.raw,
  trControl=my_control,
  methodList=c("glm","rpart","nnet","ranger","xgbTree","gbm"),
  metric="ROC"
)

#Train model as a meta-model, and caretEnsemble will make a robust linear combination of models using a glm.

xgb_ensemble <- caretStack(
  model_list,
  method="glm",
  metric="Sens",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)

library(caret)

confusionMatrix(as.factor(ifelse(predict(xgb_ensemble,newdata = train.raw[,1:30],type = "prob") > 0.5,'Bad','Good')),  train.raw$Class, mode = "prec_recall",positive = "Bad")

confusionMatrix(as.factor(ifelse(predict(xgb_ensemble,newdata = test.raw,type = "prob") > 0.5,'1','0')), as.factor(test.raw.class$Class), mode = "prec_recall",positive = "1")

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

saveRDS(xgb_ensemble, "C:/Users/Mark/Documents/creditcard/xgb_subset_ensemble.rds")
