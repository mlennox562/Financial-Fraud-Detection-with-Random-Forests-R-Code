#Timing the entire procedure.

start.time <- Sys.time()

x<-c("DMwR", "caret", "unbalanced")

lapply(x, require, character.only = TRUE)

train.raw <- read.csv("C:/Users/Mark/Documents/creditcard/train.csv",stringsAsFactors = FALSE)

train_id <- train.raw$id

#train.raw$Class <- as.factor(train.raw$Class)

#train.raw$Class <- as.factor(ifelse(train.raw$Class == 0,'Good', 'Bad'))

train.raw <- train.raw[,-1]

train.raw <- train.raw[,-1]
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

n <- 20

m <- round(n/2)

train_tomek_smote <- list()

set.seed(1234)

for (i in c(1:n)) {
  ## SMOTE after tomek links 
  
  train_tomek_smote[[i]] <- SMOTE(Class ~ ., train_tomek, perc.over = 100,perc.under = 100*sample(1:50,1))
  cat("(",table(train_tomek_smote[[i]]$Class),") ")
}

# for (i in c((m+1):n)) {
#   ## SMOTE after tomek links 
#   
#   train_tomek_smote[[i]] <- SMOTE(Class ~ ., train_tomek, perc.over = 100,perc.under = 100*sample(20:50,1))
#   cat("(",table(train_tomek_smote[[i]]$Class),") ")
# }

names(train_tomek_smote) <- paste("train_tomek_smote_",seq(1,n,length=n), sep = "")

for (i in c(1:n)) {
  lapply(1:length(train_tomek_smote), function(i) write.csv(train_tomek_smote[[i]], 
                                                            file = paste0("C:/Users/Mark/Documents/creditcard/smote_samples/",names(train_tomek_smote[i]), ".csv", sep=""),
                                                            row.names = FALSE))
}

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
