#The training and testing datasets are read into R and then prepared for the model building phase.

library(doParallel)
library(caret)
library(xgboost)

train_data <- read.csv('C:/Users/Mark/Documents/creditcard/train.csv')

test_data <- read.csv('C:/Users/Mark/Documents/creditcard/test_NoClass.csv')

test_data_Class <- read.csv('C:/Users/Mark/Documents/creditcard/test_data_Class.csv')

train_data <- train_data[,-1]

test_data <- test_data[,-1]

test_data_Class <- test_data_Class[,-1]

train_data_Class <- train_data[,32]

str(train_data)

colnames(train_data)

colnames(test_data)

colnames(test_data_Class)

train_data <- train_data[,-1]

test_data <- test_data[,-1]

test_data_Class <- test_data_Class[,-1]

train_data$Class[train_data$Class==0] <- 'Good'
train_data$Class[train_data$Class==1] <- 'Bad'

train_data_Class[train_data_Class==0] <- 'Good'
train_data_Class[train_data_Class==1] <- 'Bad'

test_data_Class[test_data_Class==0] <- 'Good'
test_data_Class[test_data_Class==1] <- 'Bad'

'========================================================================================================='

## Extreme Gradient Boosting Model  

# Set up training control
ctrl <- trainControl(method = "repeatedcv",  
                     number = 5,							# do 5 repititions of cv
                     summaryFunction=twoClassSummary,	
                     classProbs=TRUE,
                     allowParallel = TRUE)

# Use the expand.grid to specify the search space	
# Note that the default search grid selects multiple values of each tuning parameter

grid <- expand.grid(nrounds=10000, 
                    max_depth=10, 
                    eta=0.03, 
                    gamma=0.1, 
                    colsample_bytree=0.4, 
                    min_child_weight=1)
#											
set.seed(1951)  # set the seed

# Set up to do parallel processing   
registerDoParallel(4)		# Registrer a parallel backend for train
getDoParWorkers()

#Timing the model building phase.

start.time <- Sys.time()

gbm.tune <- train(x=train_data[,-31],y=train_data$Class,
                  method = "xgbTree",
                  metric = "ROC",
                  trControl = ctrl,
                  tuneGrid=grid,
                  verbose=FALSE)

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

#Making Predictions on the test and train data

train_pred <- predict(gbm.tune, train_data[,-31], type = "raw")

test_pred <- predict(gbm.tune, test_data, type = c("raw"))

'========================================================================================================='

table(test_data_Class)

library(caret)

#Obtaining F1, recall and precision scores for the model 

confusionMatrix(train_pred, train_data_Class, mode = "prec_recall",positive = "Bad")

confusionMatrix(test_pred, test_data_Class, mode = "prec_recall",positive = "Bad")

