
library(plyr)
library(dplyr)
library(caret)
library(Boruta)
library(ggplot2)
?read.csv

##read in the training and test data

train<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
names(train)
test<-read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
names(test)

##

# test subset: 30% of global training data
TrainTestIndex <- createDataPartition(y = train$classe, p = 0.3, list = F)
TrainTest <- train[TrainTestIndex, ]

# training subset: 70% of original training data will be used for exploraotry analysis and trianing the model
train <- train[-TrainTestIndex, ]
nrow(train)
ncol(train)
summary(train)
##Exploratory data analysis
##check for class imbalance
c <- ggplot(train, aes(classe))

# By default, uses stat="bin", which gives the count in each category
c + geom_bar()+ggtitle("counts of each class")


##get how many rows

by_classe <- group_by(train, classe)
classe_count <- summarise(by_classe,
                   count = n())
print(classe_count)

# classe count
# 1      A  2731
# 2      B  1867
# 3      C  1678
# 4      D  1580
# 5      E  1768
# > 

##count by class using dplyr


##clean the data

trainingnas <- apply(train, 2, function(x) {
  sum(is.na(x))
})

str(trainingnas)

##remove rows with na's
trainv2<- train[, which(trainingnas == 0)]
summary(trainv2)

##remove rows with Zero variance
nzvartrainv3 <- nearZeroVar(trainv2, saveMetrics= TRUE)
##http://topepo.github.io/caret/preprocess.html

trainv3 <- trainv2[,-nearZeroVar(trainv2)]

nrow(trainv3)
?str
str(trainv3,list.len=ncol(trainv3)) ##59 rows
##rows one and two are removed as these are index like columns
##remove X and user_name as these are just indexes

trainv4<-trainv3[,3:59]
str(trainv4)
##remove the following as these are useless attributes
# $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 440390 484434 500302 528316 560359 ...
# $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
# $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
ncol(trainv4)
colnames(trainv4)
trainv4<-trainv4[,5:57]
colnames(trainv4)
ncol(trainv4)
##find highly correlated variabales
?cor
##remove highly correlated variables
ncol(trainv4)
descrCor <- cor(trainv4[,1:52])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
trainv5 <- trainv4[,-highlyCorDescr]
str(trainv5)


library(Hmisc)
chosen_variables<-varclus(classe~.,data=trainv5)

plot(varclus(classe~.,data=trainv5)) ##use a hierarchical clustering algo to pick out indiv

##what varclus picks as the differnet groups


summary(chosen_variables)

chosen_variables$hclust

colnames(trainv5)

trainv6<-dplyr::select(trainv5,
              gyros_dumbbell_y, 
              magnet_forearm_y,
              gyros_arm_x, 
              gyros_forearm_x,
              magnet_arm_x,
              pitch_arm,
              magnet_belt_z,
              roll_dumbbell,
              pitch_dumbbell,
              total_accel_dumbbell,
              magnet_belt_x,
              roll_forearm,
              yaw_arm,
              gyros_belt_x,
              roll_arm,
              gyros_arm_z,
              yaw_belt,
              gyros_belt_z,
              total_accel_arm,
              pitch_forearm,
              accel_forearm_x,
              classe
              )


#prep <- preProcess(trainv6[, numeric_cols], method = c("center","scale","medianImpute"))

trControl <- trainControl(method = "cv", number = 10)

# build model
modelFit.rf <- train(trainv6$classe ~ ., method = "rf", trControl = trControl, 
                     trainv6)

summary(modelFit.rf)

print(modelFit.rf)

# Random Forest 
# 
# 13733 samples
# 21 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# 
# Summary of sample sizes: 12361, 12359, 12360, 12360, 12358, 12359, ... 
# 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
# 2    0.9879845  0.9847990  0.003405560  0.004309922
# 11    0.9856541  0.9818501  0.002939442  0.003719574
# 21    0.9772081  0.9711685  0.005472226  0.006925130
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 2. 


##use boruta to pick the most usefull samples

##Estimate out of sample error

##The  sample error rate is 1.3% ## (1 - .987 = 0.013 * 100).

##Estimate out of sample error

##now compare to C5.0 model

trControl <- trainControl(method = "cv", number = 10)

# build model
modelFit.C5.0 <- train(trainv6$classe ~ ., method = "C5.0", trControl = trControl, 
                    trainv6)

print(modelFit.C5.0)

# > print(modelFit.C5.0)
# C5.0 
# 
# 9624 samples
# 21 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# 
# Summary of sample sizes: 8662, 8663, 8661, 8661, 8661, 8662, ... 
# 
# Resampling results across tuning parameters:
#   
#   model  winnow  trials  Accuracy   Kappa      Accuracy SD  Kappa SD   
# rules  FALSE    1      0.9240440  0.9039599  0.010004566  0.012651181
# rules  FALSE   10      0.9731907  0.9660858  0.005346488  0.006768714
# rules  FALSE   20      0.9792174  0.9737138  0.005696525  0.007208095
# rules   TRUE    1      0.9237325  0.9035688  0.010119036  0.012793488
# rules   TRUE   10      0.9724638  0.9651690  0.004786711  0.006055907
# rules   TRUE   20      0.9793213  0.9738455  0.005426449  0.006865546
# tree   FALSE    1      0.9131339  0.8901114  0.011826512  0.014942420
# tree   FALSE   10      0.9732941  0.9662235  0.007949453  0.010053271
# tree   FALSE   20      0.9768273  0.9706926  0.006182900  0.007815163
# tree    TRUE    1      0.9135492  0.8906402  0.011722808  0.014812137
# tree    TRUE   10      0.9745402  0.9678001  0.007079081  0.008948216
# tree    TRUE   20      0.9770350  0.9709549  0.006493493  0.008208880
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were trials = 20, model = rules and winnow = TRUE.

trControl <- trainControl(method = "cv", number = 10)

modelFit.nb <- train(classe ~.,data=trainv6,
                  preProcess=c("center","scale"),method="nb", trControl = trControl)

print(modelFit.nb)

trControl <- trainControl(method = "cv", number = 10)

modelFit.svm <- train(classe ~.,data=trainv6,
                     preProcess=c("center","scale"),method="svmRadial", trControl = trControl)

print(modelFit.svm)

trControl <- trainControl(method = "cv", number = 10)

str(trainv6)
?caret::train
modelFit.nnet <- train(classe ~.,data=trainv6,
                      preProcess=c("center","scale"),method="nnet",trControl = trControl)


print(modelFit.nnet)

modelFit.nnet.unproc <- train(classe ~.,data=trainv6,method="nnet",trControl = trControl)


print(modelFit.nnet.unproc)


trControl <- trainControl(method = "cv", number = 10)

str(trainv6)

?caret::train

modelFit.knn <- train(classe ~.,data=trainv6,
                       preProcess=c("center","scale"),method="knn",trControl = trControl)

print(modelFit.knn)

modelFit.gbm <- train(classe ~.,data=trainv6,
                      preProcess=c("center","scale"),method="gbm",trControl = trControl)

print(modelFit.gbm)

modelFit.gbm.unproc <- train(classe ~.,data=trainv6,method="gbm",trControl = trControl)

print(modelFit.gbm.unproc)

##put all models into the list below and plot
results <- resamples(list(Random_Forest=modelFit.rf,C50=modelFit.C5.0,
                          nb=modelFit.nb,svm=modelFit.svm,NNET_Unprocessed=modelFit.nnet.unproc,
                          NNET_Processed=modelFit.nnet,
                          knn=modelFit.knn , 
                          GBM_Processed=modelFit.gbm , GBM_Unprocessed=modelFit.gbm.unproc))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)




###
## filter on the selected covariates identified by the feature selection techniques
##on test data
str(TrainTest)

TrainTestSbset<-dplyr::select(TrainTest,
                           gyros_dumbbell_y, 
                           magnet_forearm_y,
                           gyros_arm_x, 
                           gyros_forearm_x,
                           magnet_arm_x,
                           pitch_arm,
                           magnet_belt_z,
                           roll_dumbbell,
                           pitch_dumbbell,
                           total_accel_dumbbell,
                           magnet_belt_x,
                           roll_forearm,
                           yaw_arm,
                           gyros_belt_x,
                           roll_arm,
                           gyros_arm_z,
                           yaw_belt,
                           gyros_belt_z,
                           total_accel_arm,
                           pitch_forearm,
                           accel_forearm_x,classe
                           )

predictions <- predict(modelFit.rf, TrainTestSbset)
# length of the predictions
length(predictions)
predictions
# true accuracy of the predicted model

tsetclasse<-TrainTestSbset$classe

outOfSampleError.accuracy <- sum(predictions == TrainTestSbset$classe)/length(predictions)

outOfSampleError.accuracy
##[1] 0.9901511
# out of sample error and percentage of out of sample error
## wrap up in a function
outOfSampleError <- 1 - outOfSampleError.accuracy
print(outOfSampleError)
e <- outOfSampleError * 100
paste0("The Out of sample error is estimated to be: ", round(e, digits = 2), "%")

##Out of sample error function

out.of.sample.error.function<-function(model,trainset,trainset.class){
  
  predictions <- predict(model, trainset)
  
  outOfSampleError.accuracy <- sum(predictions == trainset.class)/length(predictions)
  
  outOfSampleError <- 1 - outOfSampleError.accuracy
  
  return(outOfSampleError*100)
}


out.of.sample.error.function(modelFit.rf,TrainTestSbset,TrainTestSbset$classe)

library(Boruta)

Bor.trainv5<-Boruta(classe~.,data=trainv5)

str(trainv5)

summary(Bor.trainv5)

Bor.trainv5$finalDecision
##use carets forward selection algo
##no reduction
Bor.trainv5$finalDecision
# yaw_belt         gyros_belt_x         gyros_belt_y         gyros_belt_z        magnet_belt_x 
# Confirmed            Confirmed            Confirmed            Confirmed            Confirmed 
# magnet_belt_z             roll_arm            pitch_arm              yaw_arm      total_accel_arm 
# Confirmed            Confirmed            Confirmed            Confirmed            Confirmed 
# gyros_arm_x          gyros_arm_z          accel_arm_y         magnet_arm_x         magnet_arm_z 
# Confirmed            Confirmed            Confirmed            Confirmed            Confirmed 
# roll_dumbbell       pitch_dumbbell         yaw_dumbbell total_accel_dumbbell     gyros_dumbbell_y 
# Confirmed            Confirmed            Confirmed            Confirmed            Confirmed 
# gyros_dumbbell_z    magnet_dumbbell_z         roll_forearm        pitch_forearm          yaw_forearm 
# Confirmed            Confirmed            Confirmed            Confirmed            Confirmed 
# total_accel_forearm      gyros_forearm_x      accel_forearm_x      accel_forearm_z     magnet_forearm_x 
# Confirmed            Confirmed            Confirmed            Confirmed            Confirmed 
# magnet_forearm_y     magnet_forearm_z 
# Confirmed            Confirmed 
# Levels: Tentative Confirmed Rejected
# > 


control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run the RFE algorithm

results <- rfe(trainv5[,1:25], trainv5[,26], sizes=c(1:25), rfeControl=control)

# summarize the results

print(results)
predictors(results)
plot(results, type=c("a"))



# Recursive feature selection
# 
# Outer resampling method: Cross-Validated (10 fold) 
# 
# Resampling performance over subset size:
  
#   Variables Accuracy  Kappa AccuracySD  KappaSD Selected
# 1   0.9923 0.9903  0.0020153 0.002549         
# 2   0.9805 0.9753  0.0094871 0.012028         
# 3   0.9808 0.9757  0.0110494 0.013963         
# 4   0.9959 0.9948  0.0021613 0.002733         
# 5   0.9971 0.9963  0.0018915 0.002392         
# 6   0.9966 0.9957  0.0015967 0.002020         
# 7   0.9968 0.9959  0.0020105 0.002543         
# 8   0.9967 0.9959  0.0017677 0.002236         
# 9   0.9979 0.9973  0.0012663 0.001602         
# 10   0.9979 0.9973  0.0014163 0.001791         
# 11   0.9976 0.9970  0.0014617 0.001849         
# 12   0.9976 0.9970  0.0019518 0.002469         
# 13   0.9977 0.9971  0.0018483 0.002338         
# 14   0.9977 0.9971  0.0011576 0.001464         
# 15   0.9981 0.9976  0.0009917 0.001254         
# 16   0.9982 0.9977  0.0010795 0.001365         
# 17   0.9981 0.9976  0.0009619 0.001217         
# 18   0.9983 0.9978  0.0010792 0.001365         
# 19   0.9984 0.9979  0.0011716 0.001482         
# 20   0.9982 0.9977  0.0011069 0.001400         
# 21   0.9984 0.9979  0.0011716 0.001482        *
# 22   0.9984 0.9979  0.0012665 0.001602         
# 23   0.9982 0.9977  0.0013838 0.001750         
# 24   0.9983 0.9978  0.0012525 0.001584         
# 25   0.9981 0.9976  0.0011277 0.001426         
# 
# The top 5 variables (out of 21):
#   raw_timestamp_part_1, yaw_arm, roll_arm, gyros_belt_z, magnet_forearm_x


##all variables appear to be importnat from both Boruta and Forward selection

ncol(train)

featurePlot(x = trainv5[, 1:25], 
            y = trainv5$classe, 
            plot = "pairs"
            ## Add a key at the top 
)


##make predictions





testtestset<-dplyr::select(test,
                              gyros_dumbbell_y, 
                              magnet_forearm_y,
                              gyros_arm_x, 
                              gyros_forearm_x,
                              magnet_arm_x,
                              pitch_arm,
                              magnet_belt_z,
                              roll_dumbbell,
                              pitch_dumbbell,
                              total_accel_dumbbell,
                              magnet_belt_x,
                              roll_forearm,
                              yaw_arm,
                              gyros_belt_x,
                              roll_arm,
                              gyros_arm_z,
                              yaw_belt,
                              gyros_belt_z,
                              total_accel_arm,
                              pitch_forearm,
                              accel_forearm_x)

predictions <- predict(modelFit.rf, testtestset)

print(predictions)

# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E