library(unbalanced)

#read the training data into R

train_data <- read.csv('/users/mlennox05/Documents/creditcard/train.csv')

train_data <- train_data[,-1]

table(train_data$Class)

n<-ncol(train_data)
output<-train_data$Class
input<-train_data[ ,-n]

#run the ENN algorithm within the unbalanced package.

data<-ubENN(X=input, Y= output)
newData<-cbind(data$X, data$Y)

table(newData$`data$Y`)

colnames(newData) <- c("id","Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class")

#write the new training dataset to a csv file 

write.csv(newData, '/users/mlennox05/Documents/creditcard/balancedData_ENN.csv')
