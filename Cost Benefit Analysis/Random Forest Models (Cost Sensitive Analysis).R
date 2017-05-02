library(mlr)
library(parallelMap)
parallelStartSocket(3)

'========================================================================================================='

test_data <- read.csv('/users/mlennox05/Documents/creditcard/test.csv')

test_data <- test_data[,-1]

test_data <- test_data[,-1]

test_data$Class[test_data$Class==0] <- 'Good'

test_data$Class[test_data$Class==1] <- 'Bad'

test.task = makeClassifTask(data = test_data, target = "Class",positive = "Bad")

c_a <- 1

costs = matrix(c(c_a, c_a, 100*c_a, 0), 2)

colnames(costs) = rownames(costs) = getTaskClassLevels(test.task)

credit.costs = makeCostMeasure(id = "credit.costs", name = "Credit costs", costs = costs,
                               best = 0, worst = 100*c_a)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_Under.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

th = costs[2,1]/(costs[2,1] + costs[1,2] - costs[1,1] - costs[2,2])

# lrn = makeLearner("classif.randomForest", predict.type = "prob")

rin = makeResampleInstance("CV", iters = 3, task = train.task)
lrn = makeLearner("classif.randomForest", predict.type = "prob", predict.threshold = th)
r = resample(lrn, train.task, resampling = rin, measures = list(credit.costs, mmce), show.info = FALSE)

# mod = train(lrn, train.task)
# pred = predict(mod, task = test.task)

cat("Under:") 
performance(setThreshold(r$pred, 0.5), measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_SMOTE.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("SMOTE:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_CNN.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("CNN:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_ENN.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("ENN:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_Tomek.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("Tomek:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_OSS.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("OSS:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_NCL.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("NCL:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/train.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("No Re-Balancing:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

train_data <- read.csv("/users/mlennox05/Documents/creditcard/balancedData_Over.csv")

train_data <- train_data[,-1]

train_data <- train_data[,-1]

train_data$Class[train_data$Class == 1] <- 'Bad'

train_data$Class[train_data$Class == 0] <- 'Good'

train.task = makeClassifTask(data = train_data, target = "Class",positive = "Bad")

lrn = makeLearner("classif.randomForest", predict.type = "prob")
mod = train(lrn, train.task)
pred = predict(mod, task = test.task)

cat("Over:") 
performance(pred, measures = list(credit.costs, mmce))

rm(train_data)

'========================================================================================================='

parallelStop()
