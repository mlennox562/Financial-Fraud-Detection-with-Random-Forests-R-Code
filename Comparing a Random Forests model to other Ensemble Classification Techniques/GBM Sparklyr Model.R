#The training and testing datasets are read into R and then prepared for the sparklyr model building phase.

library(randomForest)
library(sparklyr)
library(dplyr)
library(ggplot2)

#connecting to the spark cluster

sc <- spark_connect(master = "local")

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

#Creating a spark data-table for both the training and testing datasets.  

train_data_tbl <- copy_to(sc, train_data, "train_data", overwrite = TRUE)%>% 
  mutate(Class = as.character(Class))

test_data_tbl <- copy_to(sc, test_data, "test_data", overwrite = TRUE)

'========================================================================================================='

#Timing the model building phase.

start.time <- Sys.time()

GBM_model <- train_data_tbl %>%
  ml_gradient_boosted_trees(
    response = "Class",
    features = colnames(train_data[,-31]),
    num.trees = 500L,
    type = c("classification")
  )

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

#Making Predictions on the test and train data

train_pred <- predict(GBM_model, train_data_tbl)

test_pred <- predict(GBM_model, test_data_tbl)

'========================================================================================================='

table(test_data_Class)

library(caret)

#Obtaining F1, recall and precision scores for the model 

confusionMatrix(train_pred, train_data_Class, mode = "prec_recall",positive = "1")

confusionMatrix(test_pred, test_data_Class, mode = "prec_recall",positive = "1")

