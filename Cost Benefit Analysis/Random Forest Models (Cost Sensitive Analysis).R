library(mlr)
library(parallelMap) #Unified parallelization framework for multiple back-end, designed for internal package and interactive usage. 
parallelStartSocket(3)

'========================================================================================================='

#read the testing data into R

test_data <- read.csv('/users/mlennox05/Documents/creditcard/test.csv')

test_data <- test_data[,-1]

test_data <- test_data[,-1]

test_data$Class[test_data$Class==0] <- 'Good'

test_data$Class[test_data$Class==1] <- 'Bad'

test.task = makeClassifTask(data = test_data, target = "Class",positive = "Bad")

#Let c_a denote the cost of analysis

c_a <- 1

#build a cost matrix 

costs = matrix(c(c_a, c_a, 100*c_a, 0), 2)

colnames(costs) = rownames(costs) = getTaskClassLevels(test.task)

#In order to calculate the average costs over the entire data set we first need to create a new performance Measure using makeCostMeasure

credit.costs = makeCostMeasure(id = "credit.costs", name = "Credit costs", costs = costs,
                               best = 0, worst = 100*c_a)

'========================================================================================================='

#read the under-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_Under.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#threshold

th = costs[2,1]/(costs[2,1] + costs[1,2] - costs[1,1] - costs[2,2])

# lrn = makeLearner("classif.randomForest", predict.type = "prob")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

rin = makeResampleInstance("CV", iters = 3, task = train.task)
lrn = makeLearner("classif.randomForest", predict.type = "prob", predict.threshold = th)
r = resample(lrn, train.task, resampling = rin, measures = list(credit.costs, mmce), show.info = FALSE)

# mod = train(lrn, train.task)
# pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("Under:") 
performance(setThreshold(r$pred, 0.5), measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the SMOTE-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_SMOTE.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("SMOTE:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the CNN-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_CNN.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("CNN:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the ENN-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_ENN.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("ENN:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the Tomek_link-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_Tomek.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("Tomek:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the OSS-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_OSS.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("OSS:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the NCL-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_NCL.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("NCL:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/train.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("No Re-Balancing:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

#read the Over-sampled training dataset into R

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_Over.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

#In the R code below we make use of the predict.threshold argument of makeLearner to set the threshold before doing a 3-fold cross-validation on the credit.task. 

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

#Then the average costs can be computed by function performance. Below we compare the average costs and the error rate (mmce)

cat("Over:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

parallelStop()
