require(dplyr)
library(stats)
library(Metrics)
library(glmnet)
library(ggplot2)
library(caret)
library(car)
HP.train <- read.csv('train.csv', sep = ',', header = TRUE)
HP.test <- read.csv('test.csv', sep = ',', header = TRUE)
HP.train$Id <- NULL
HP.test.Id <- HP.test$Id
HP.test$Id <- NULL
HP.test$SalePrice <- replicate(nrow(HP.test), -1)
ntrain <- nrow(HP.train)
ntest <- nrow(HP.test)
HP.all <- rbind(HP.train, HP.test)
check_missing_values <- function(dataSet)
{
  for (i in seq(1,ncol(dataSet))) {
    num_NA = sum(is.na(dataSet[, i]))
    is_numeric = is.numeric(dataSet[,i]) 
    if (num_NA > 0) {
      print (paste0('column ', names(dataSet)[i],' has #missing values: ', num_NA, ' is numeric: ',is_numeric ))
    }
  }
}
check_categorical_variables <- function(dataSet)
{
  print('************************************************')
  print('Listing all variables with categorical values:')
  count <- 0
  for (i in 1:ncol(dataSet)) {
    if (!is.numeric(dataSet[,i])) {
      count <- count + 1
      factor_names <- paste(levels(dataSet[,i]), sep=" ")
      print (paste0(names(dataSet)[i], ' is categorical variable with number of factors: '))
      print (factor_names)
    }
  }
  print (paste0('Total categorical variables:' , count))
  print('************************************************')
}

deal_missing_values <- function(dataSet, ntrain = 1460, type = 1)
{
  # type is the replace method for numeric values. 
  #if type = 1 --> using mean to replace.
  #if type = 2 --> using median to replace.
  dataSet[] <- lapply(dataSet, function(x){
    # check if variables have a factor:
    if(!is.factor(x)) {
      #replace NA by mean
      if (type == 1) x[is.na(x)] <- mean(x[1:1460], na.rm = TRUE) 
      else if (type == 2) if (type == 1) x[is.na(x)] <- median(x[1:1460], na.rm = TRUE) 
    }
    else {
      # otherwise include NAs into factor levels and change factor levels:
      x <- factor(x, exclude=NULL)
      levels(x)[is.na(levels(x))] <- "Missing"
    }
    return(x)
  })
  
  return(dataSet)
}

#
# encoding categorical variables. If a variable has 3 levels --> we need 2 bit to encode
#
encode_categorical_variables <- function(dataSet, ntrain=1460)
{
  dataSetTemp <- dataSet
  for (i in 1:ncol(dataSet)){
    if (is.factor(dataSet[,i])) {
      col_name = names(dataSet)[i]
      t <- model.matrix(~ factor(dataSet[,i]) - 1)
      dataSetTemp <- cbind(t[,2:ncol(t)], dataSetTemp)
      new_col_names <- paste(col_name, "_bit", seq(1, ncol(t) - 1))
      names(dataSetTemp)[1:ncol(t) - 1] <- new_col_names
    }
  }
  removed <- c()
  for (i in 1:ncol(dataSetTemp)) {
    if (is.factor(dataSetTemp[,i])) removed <- c(removed, i)
  }
  dataSetTemp <- dataSetTemp[, -removed]
  return(dataSetTemp)
}

deal_outliers <- function(dataSet)
{
  
}

do_variable_selection <- function(dataSet)
{
  #must return a dataframe in which important variables are kept.
    
}

build_model <- function(train, type = 2)
{
  if (type == 1) {
    #multiple linear regression model
    fit <- lm(SalePrice ~., train)  
    fit$rsquared <- summary(fit)$r.squared
    fit$adj.rsquared <- summary(fit)$adj.r.squared
    fit$predicted <- fit$fitted.values
    
  }
  if (type == 2) {
    #add elastic net regularization to reduce model overfitting
    # fit = cv.glmnet(x = as.matrix(train[,1:(ncol(train) - 1)]), y=as.matrix(train[,ncol(train)]), type.measure = "mse")
    # plot(fit)
    set.seed(1024)
    fit = glmnet(x = as.matrix(train[,1:(ncol(train) - 1)]), y=as.matrix(train[,ncol(train)]), family="gaussian", alpha = 0.1, lambda = 0.1)
    # plot(fit, xvar = "dev", label = TRUE)
    # plot(fit)
    # print(fit$lambda.min)
    # print(fit$lambda.lse)
    fit$predicted <- predict(fit, newx = as.matrix(train[,1:(ncol(train) - 1)]),  type="response", family="gaussian", standardize=TRUE)
  }
  if (type == 3) {
    # random forest in here:
    fit <- randomForest(x = as.matrix(train[,1:(ncol(train) - 1)]), y = as.matrix(train[,ncol(train)]), 
                        ntree = 200, mtry = 100)
    plot(fit)
    # fit$predicted <- as.vector(fit$predicted)
  }
  fit$actual <- train$SalePrice
  fit$rmse <- rmse(actual = fit$actual, predicted  = fit$predicted)
  print(paste0('RMSE of training :',fit$rmse))
  return(fit)
}
do_transform <- function(dataSet)
{
  for (i in 1:(ncol(dataSet) - 1) ) {
    if (min(dataSet[,i]) == 0 && max(dataSet[,i]) == 1 ) {
      #categorical values --> do nothing
    } else{
      dataSet[,i] <- log(dataSet[,i] + 1)
    }
  }
  return(dataSet)
  
}
evaluate_model <- function(model, test)
{
  eval.predicted <- predict(model, data=test)
  eval.actual <- test$SalePrice
  eval.RMSE <- rmse(actual = eval.actual, predicted = eval.predicted)
  
}


#missing values: categorical + numeric values. Numeric values can be replaced by mean, median for ease.
#outliers: do later.
#do regression: variable selection + linear regression + xgboost + random forest
# check the residuals
# Mine: regression part + transformation 

####################### 1. Dealing with missing values for training and testing data #################################################
check_missing_values(HP.all)
HP.all <- deal_missing_values(HP.all, ntrain)
print ('check missing values of the dataframe after replacing missing values. Should print blank')
check_missing_values(HP.all)

#***************************************************************************************************************
###############2. Linear regression model with untransformed sale price data and residual analysis ######################
#***************************************************************************************************************

HP.train<-deal_missing_values(HP.train)

do_recode_nas_to_none<-function(dataSet=HP.train) {
nas<-which(is.na(dataSet)==TRUE)

data.cat<-as.matrix(dataSet)
data.temp[nas]<-"none"
data.recoded<-as.data.frame(data.temp)

return(data.recoded)
}

HP.train1<-do_recode_nas_to_none()

HP.train.lr_model <- lm(SalePrice~., HP.train)


### Residual Plots

#Residual vs Fitted Plot
plot(HP.train.lr_model, which=1)
#There's a pretty clear expansion in variance once the 
# sale price goes above 3e+05. Makes sense, since there 
#are fewer homes at that price range.
#The base r plots point out some outliers too, obs. # 826, 1325, 524

summary(HP.train.lr_model)
#QQ plot
plot(HP.train.lr_model, which=2)

#QQ plot looks pretty bad at the tails. 
#826 and 524 are again noted as outliers

#fitted values by square root of standardized residuals
plot(HP.train.lr_model, which=3)

#error clearly increases at higher ends of sale price

#cook's distance
plot(HP.train.lr_model, which=4)
#3 outliers in terms of cook's distance: 524, 1171, and 1424

#Leverage vs. standardized residuals
plot(HP.train.lr_model, which=5)
#observations with high leverage and high standardized residuals 
# are the same that have high cook's d

#We have five clear outliers that need looking at. Log transform should help.
#These are: 524,826,1171,1325, 1424

build_model(HP.train, type = 1)


#Since I have the linear model code up and running here and Thanh already made all this 
#lovely code for us, I'll just run it with the log-transform on sale price. 

#***************************************************************************************************************
###############3. Linear regression model with log transform of sale price data and residual analysis ######################
#***************************************************************************************************************


HP.trainlog<-HP.train %>% mutate(logSalePrice=log(SalePrice)) %>% select (-SalePrice)

HP.train_logtrans_mod<-lm(logSalePrice ~., HP.trainlog)

#Same Plots again

#outliers from untransformed were 524,826,1171,1325, 1424

#residuals vs fitted
plot(HP.train_logtrans_mod, which=1, id.n=5)
# Residuals over .5 also now include observations 633 and 463


#An alternate method for seeing the labels for all residuals, since r only prints
#a given number of extreme values
# logtransmod<-data.frame(HP.train_logtrans_mod$residuals,
#                         HP.train_logtrans_mod$fitted.values,
#                         HP.trainlog$logSalePrice)
# colnames(logtransmod)<-c("residuals","predicted","logSalePrice")
# logtransmod$obs<-as.character(1:1460)
# 
# logtransmod %>% 
# ggplot(aes(x=predicted,y=residuals, label=obs))+geom_text()


plot(HP.train_logtrans_mod, which=2, id.n=6)

#observation 89 shows up as overestimating the price based on the trend line
#of the quantiles. The QQ plot for this model does not look much better than previous

plot(HP.train_logtrans_mod, which=3)
#The scale-location plot looks far better for this model, although error is slightly 
#higher at the lower end now. The same outliers are identified

plot(HP.train_logtrans_mod, which=4)
#89 is now identified as an influential observation by cook's distance

plot(HP.train_logtrans_mod, which=5)
#Only a few observations with high leverage and high standardized residuals, which
#we already knew about.

summary(HP.train_logtrans_mod)

summary(HP.train.lr_model)

#Comparing with the untransformed model, we get a higher adjusted R-squared and a higher
#F statistic when predicting the log transform of sale price. 

#************************************************************************************************************************************#
####################### 2. Encoding categorical variables ############################################################################
check_categorical_variables(HP.train)
# HP.train <- encode_categorical_variables(HP.train)
HP.all <- encode_categorical_variables(HP.all)
print ('check categorical values of the dataframe after encoding. Should print blank')
check_categorical_variables(HP.all)
#************************************************************************************************************************************#
#do transformation on variables?
HP.all <- do_transform(HP.all)


HP.train = HP.all[1:ntrain,]
HP.test = HP.all[(ntrain+1) : nrow(HP.all),]
HP.test$SalePrice <- NULL


print ('Building multiple linear regression model')
HP.train.lr_model <- build_model(HP.train, type = 1)





print ('Building glmnet model')
HP.train.glmnet_model <- build_model(HP.train, type = 2)
print ('Building random forest model')
HP.train.rf_model <- build_model(HP.train, type = 3)

print('Choosing?')
HP.test$SalePrice <- predict(HP.train.rf_model, as.matrix(HP.test))

#write to file
submission <- data.frame(Id <- HP.test.Id, SalePrice <- HP.test$SalePrice)
names(submission) <- c('Id', 'SalePrice')
write.csv(file = 'submission.csv', x = submission, row.names = FALSE)
