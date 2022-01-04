library(xgboost)
library(caret)
library(dplyr)
library(tidyverse)
library(vtreat)
library(ggplot2)
library(partykit)
library(precrec)

data<-read.csv("~/maintenance.csv")

data<-data[,c(-1,-2)]                                   # remove column 1 and 2
data$Type<-as.factor(data$Type)                         # change to factor
data<-data %>%                                          #rename columns
  rename(Airtemp = Air.temperature..K.,
         Proctemp = Process.temperature..K.,
         Rotationalspeed = Rotational.speed..rpm.,
         Torque = Torque..Nm.,
         Toolwear = Tool.wear..min.,
         Failure = Machine.failure) 

# TRAIN TEST SPLIT 70%
set.seed(200)
trainRowNumbers <- createDataPartition(data$Failure, p=0.7, list=FALSE)
trainData <- data[trainRowNumbers,]
testData <- data[-trainRowNumbers,]

# Check data imbalance
table(trainData$Failure)
table(testData$Failure)

########################
######CTREE######
set.seed(400)
ctree_model =  ctree(as.factor(Failure) ~ ., data = trainData[,-8:-11], 
                     control = ctree_control(testtype = "Bonferroni", alpha = 0.44))
#plot 
plot(ctree_model, drop_terminal = F, gp = gpar(fontsize = 8), inner_panel=node_inner)

#better plot
st <- as.simpleparty(ctree_model)
myfun <- function(i) c(
  as.character(i$prediction),
  paste("n =", i$n),
  format(round(i$distribution/i$n, digits = 3), nsmall = 2)
)
plot(st, tp_args = list(FUN = myfun), ep_args = list(justmin = 20),gp = gpar(fontsize = 7), drop_terminal = T)

#prediction test data
predctree <- predict(ctree_model, testData[,-8:-11])

#confusionmatrix of predictions on testset
cmctree<-confusionMatrix(as.factor(predctree), as.factor(testData$Failure), positive = "1")

#check auprc
ctreeauc <- evalmod(scores=as.numeric(predctree), labels=testData$Failure)
aucctree<-auc(ctreeauc)
aucctree

#make table of evaluation metrics
cmctreetable<-as.matrix(cmctree, what = "classes")
cmctreetable<-cmctreetable[c(-1:-4,-8:-11),]
cmctreetable<-as.data.frame(t(cmctreetable))

cmctreetable<-cbind(cmctreetable, as.data.frame(t(cmctree$overall[2])))
cmctreetable<-cbind(cmctreetable, aucctreetable)
colnames(cmctreetable)[5] = "AUCPR"
cmctreetable

#prediction on train data
predctree <- predict(ctree_model, trainData[,-8:-11])
confusionMatrix(as.factor(predctree), as.factor(trainData$Failure), positive = "1")

#check false negatives
fullctree<-cbind(testData, predctree)
fullctree<-subset(fullctree, Failure == 1 & predctree==0)
tablectree<-apply(fullctree[,8:12], 2, table)
tablectree
#########################
######XGBOOST######
#hot encode data, xgboost needs matrix as input
features <- setdiff(names(trainData), "Failure")
treatplan <- vtreat::designTreatmentsZ(trainData, features, verbose = FALSE)

new_vars <- treatplan %>%
  magrittr::use_series(scoreFrame) %>%        
  dplyr::filter(code %in% c("clean", "lev")) %>% 
  magrittr::use_series(varName)

#set traindata
features_train <- vtreat::prepare(treatplan, trainData, varRestriction = new_vars) %>% as.matrix()
response_train <- trainData$Failure
#rename machine type
colnames(features_train)[11] = "TypeH"
colnames(features_train)[12] = "TypeL"
colnames(features_train)[13] = "TypeM"

#set testdatas
features_test <- vtreat::prepare(treatplan, testData, varRestriction = new_vars) %>% as.matrix()
response_test <- testData$Failure
#rename machine type
colnames(features_test)[11] = "TypeH"
colnames(features_test)[12] = "TypeL"
colnames(features_test)[13] = "TypeM"

#set weight scale for imbalance parameter
sumpos<-sum(response_train)
sumneg<-sum(response_train!=1)

sumneg/sumpos

#create grid of hyperparameters for tuning 
the.grid <- expand.grid(
  eta = c(.01, .05, .1, .2, .3),
  max_depth = c(3, 6, 9, 12),
  subsample = c(.7),
  scale_pos_weight = sumneg/sumpos,
  lambda = c(1, 2, 5, 8 ,10), 
  gamma = c(0, .5, 1, 2, 3 ),
  optimal_trees = 0,               
  max_aucpr = 0                     
)

#run XGBoost CV using grid
for(i in 1:nrow(the.grid)) {
  
  # parameter list
  params <- list(
    eta = the.grid$eta[i],
    max_depth = the.grid$max_depth[i],
    scale_pos_weight = the.grid$scale_pos_weight[i],
    subsample = the.grid$subsample
  )
  

  set.seed(100)
  # train model
  xgb.tune <- xgb.cv(
    params = params,
    data = features_train[,-6:-10],
    label = response_train,
    nrounds = 5000,
    nfold = 5,
    objective = "binary:logistic",
    eval_metric = "aucpr",
    print_every_n = 20,
    verbose = 1,
    early_stopping_rounds = 700
  )
  
  # add max aucpr and trees to grid
  the.grid$optimal_trees[i] <- which.max(xgb.tune$evaluation_log$test_aucpr_mean)
  the.grid$max_aucpr[i] <- max(xgb.tune$evaluation_log$test_aucpr_mean)
}
#check results from grid after loop finished
the.grid %>%
  dplyr::arrange(desc(max_aucpr)) %>%
  head(20)

# set optimal values found in grid
params <- list(
  eta = 0.05,
  max_depth = 12,
  subsample = .7,
  scale_pos_weight = sumneg/sumpos,
  lambda = 8,
  gamma = 2 
)

# train final model using optimal values found in grid
set.seed(100)
xgb.fit.final <- xgboost(
  params = params,
  data = features_train[,-6:-10],
  label = response_train,
  nrounds = 258,
  objective = "binary:logistic",
  eval_metric = "aucpr",
  verbose = 1
)
#save as .rds for quick load
saveRDS(xgb.fit.final, "xgb.fit.final.rds")
xgb.fit.final <- readRDS("xgb.fit.final.rds")

#prediction using testdata
predxgb <- predict(xgb.fit.final, features_test[,-6:-10])
predxgb <- ifelse(predxgb > 0.5, 1, 0) #cutoff 0.5

#confusionmatrix testdata
cmxgb<-confusionMatrix(as.factor(predxgb), as.factor(testData$Failure), positive = "1")
cmxgb

#check AUPRC
xgbauc <- evalmod(scores=as.numeric(predxgb), labels=testData$Failure)
aucxgb<-auc(xgbauc)
aucxgb
aucxgbtable<-as.data.frame(aucxgb[2,4])

# evaluation metrics table
cmxgbtable<-as.matrix(cmxgb, what = "classes")
cmxgbtable<-cmxgbtable[c(-1:-4,-8:-11),]
cmxgbtable<-as.data.frame(t(cmxgbtable))
cmxgbtable<-cbind(cmxgbtable, as.data.frame(t(cmxgb$overall[2])))
cmxgbtable<-cbind(cmxgbtable, aucxgbtable)
colnames(cmxgbtable)[5] = "AUPRC"
cmxgbtable

#check false negatives
fullxgb<-cbind(testData, predxgb)
fullxgb<-subset(fullxgb, Failure == 1 & predxgb==0)

tablexgb<-apply(fullxgb[,8:12], 2, table)
tablexgb

#predictions traindata
predxgbtrain <- predict(xgb.fit.final, features_train[,-6:-10])
predxgbtrain <- ifelse(predxgbtrain > 0.5, 1, 0)
confusionMatrix(as.factor(predxgbtrain), as.factor(trainData$Failure), positive = "1")

#variable importance plot using gain
importance<- xgb.importance(model=xgb.fit.final)
xgb.ggplot.importance(importance, top_n = 10, measure = "Gain", main="Gain") + 
  ggtitle("")+ theme_minimal() + ylab("Relative Importance") + xlab("Variable")+ 
  theme(axis.title=element_text(size=9)) + theme(legend.title = element_text(size=9))+ 
  guides(fill = guide_legend(reverse=TRUE))+ 
  scale_fill_discrete(name = "Clusters", labels = c("Machine Types", "Temperatures", "Mechanics")) + 
  theme(legend.text = element_text(size = 7))

#Partial dependence plots
xgb.fit.final %>%
  partial(pred.var = "Torque", n.trees = 258, train = features_train[,-6:-10], prob = T) %>%
  autoplot(rug = TRUE, train = features_train[,-6:-10], color = "#FF9900") + theme_minimal() +
  ylab("Predicted Machine Malfuntion Probability") 

###################################
