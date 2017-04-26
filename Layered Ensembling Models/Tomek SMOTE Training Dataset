
x<-c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
     "MASS", "rpart", "gbm", "ROSE")

lapply(x, require, character.only = TRUE)

test.raw.class <- read.csv("C:/Users/Mark/Documents/creditcard/test_data_Class.csv",stringsAsFactors = FALSE)

train.raw <- read.csv("C:/Users/Mark/Documents/creditcard/train.csv",stringsAsFactors = FALSE)

train_id <- train.raw$id

#train.raw$Class <- as.factor(train.raw$Class)

#train.raw$Class <- as.factor(ifelse(train.raw$Class == 0,'Good', 'Bad'))

test.raw <- read.csv("C:/Users/Mark/Documents/creditcard/test_NoClass.csv",stringsAsFactors = FALSE)

test_id <- test.raw$id

train.raw <- train.raw[,-1]

test.raw <- test.raw[,-1]

train.raw <- train.raw[,-1]

test.raw <- test.raw[,-1]

'================================================================================================================='

################## Handling Class Imbalance with Tomek Links and Smoteing ############
########### Handling Unbalanced data by TOMEK & SMOTE ####################

## Creating tomeklinks and removing the irrelevant datapoints
set.seed(1234)
tomek = ubTomek(train.raw[,-31], train.raw[,31])
model_train_tomek = cbind(tomek$X,tomek$Y)
names(model_train_tomek)[31] = "Class"

## TOMEK Indices 
removed.index = tomek$id

# Converting Class to factor variable
train.raw$Class = as.factor(train.raw$Class)
train_tomek = train.raw[-removed.index,]

## SMOTE after tomek links 
set.seed(1234)
train_tomek_smote <- SMOTE(Class ~ ., train_tomek, perc.over = 100,perc.under = 100)
table(train_tomek_smote$Class)

write.csv(train_tomek_smote,"C:/Users/Mark/Documents/creditcard/train_tomek_smote.csv")
