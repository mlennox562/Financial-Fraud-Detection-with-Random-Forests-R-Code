# Firstly, the entire dataset is read in. 
full <- read.csv('C:/Users/Mark/Documents/Credit Card Data/creditcard.csv')

set.seed(927540672)


full_random <- full[sample(nrow(full)),] # randomize row order
train_fraction <- 0.7
train_data <- full_random[1:floor(train_fraction*nrow(full_random)),]
test_data <- full_random[(floor(train_fraction*nrow(full_random)) + 1):nrow(full_random),]
identical(full_random, rbind(train_data, test_data)) # test that split was done correctly

write.csv(train_data, 'C:/Users/Mark/Documents/Credit Card Data/train.csv')
write.csv(test_data, 'C:/Users/Mark/Documents/Credit Card Data/test.csv')

test_data_Class <- test_data[,32]
