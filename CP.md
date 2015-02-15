---
title: "Human Activity Recognition"
author: "Christian Otto"
graphics: yes
output:
  html_document:
    toc: false
    theme: united
---


# Human Activity Recognition

Course: Practical Machine Learning  
Author: Christian Otto  
Date:  February 15, 2015  


### Preparation

It is necessary to load R packages and set global options to
`knitr`. Note that it is assumed that all required packages are
installed. In addition, the random generator is initialized with the
same seed in order to obtain reproducibility of the results.


```r
library(plyr)
library(ggplot2)
library(knitr)
library(caret)
set.seed(1337)
opts_chunk$set(echo=TRUE, warning=FALSE, message=FALSE)
```

### Data Processing

#### Retrieving and Loading the Data

Download the data (if necessary) and reading the training and testing
data into a data.frame.


```r
## Set URLs accordingly
training.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
training.file <- "pml-training.csv"
testing.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testing.file <- "pml-testing.csv"

## Download files
if (!file.exists(training.file)){
   download.file(training.url, training.file, method="curl")
}
if (!file.exists(testing.file)){
   download.file(testing.url, testing.file, method="curl")
}

## Load data into R
data <- read.csv(file=training.file, na.strings=c("NA", "", "DIV/0!"))
testing <- read.csv(file=testing.file, na.strings=c("NA", "", "DIV/0!"))
```

#### Partitioning data into training and probing dataset

Partitioning the dataset into 60% training and 40% probing dataset in
order to get an estimate of the out-of-sample error.


```r
inTrain <- createDataPartition(data$classe, p = 0.6)[[1]]
training <- data[inTrain,]
probing <- data[-inTrain,]
```

#### Filtering covariates

Identify predictors in the training set that have primarily NA values
(i.e., more than 50% of the values are NA). Actually, there are 100
columns with more than 97% missing values and 60 columns with no
missing values and hence it is clear that imputation is not an
option. Moreover, identify columns are that do not contain predictor
variables. Now, the NA and non-predictor columns are removed from all
datasets (training, probing, and testing). In addition, the classe
variable is converted to a factor variable for model building. In the
end, the datasets consist of `52` covariates.


```r
NA.freq <- sapply(training, function(i){mean(is.na(i))})
NA.cols <- names(NA.freq[NA.freq > 0.5])
nonPred.cols <- c("X", "user_name", "raw_timestamp_part_1",
	     "raw_timestamp_part_2", "cvtd_timestamp",
	     "new_window", "num_window", "problem_id")
	     
training <- subset(training, select=which(!colnames(training) %in% c(NA.cols, nonPred.cols)))
probing <- subset(probing, select=which(!colnames(probing) %in% c(NA.cols, nonPred.cols)))
testing <- subset(testing, select=which(!colnames(testing) %in% c(NA.cols, nonPred.cols)))

training$classe <- factor(training$classe)
probing$classe <- factor(probing$classe)
```

#### Building models on training dataset

Here, two different prediction models (random forests and boosting)
are built on the training set since it was states in the lectures that
these overall perform very well. The parameters of the models during
training are optimized by 10-fold cross validation.


```r
ctr <- trainControl(method = "cv", number = 10)
```

```r
model.rf <- train(classe ~ ., data = training, method = "rf", trControl=ctr)
```

```r
model.boost <- train(classe ~ ., data = training, method = "gbm", verbose=FALSE, trControl=ctr)
```

Now, it is possible to get the maximum accuracy from the cross
validations even though these are not good estimates for the
out-of-sample error (therefore the probing dataset). Nevertheless,
they provide an idea about the models. It turns out that both models
are highly accurate with minor advantage for the random forest
model.


```r
data.frame(model=c("random forest", "boosting"),
           accuracy=c(round(max(model.rf$results$Accuracy), digits=3),
                      round(max(model.boost$results$Accuracy), digits=3)))
```

```
##           model accuracy
## 1 random forest    0.990
## 2      boosting    0.961
```

#### Getting estimates on out-of-sample error for both models

First, the predictions are done for the probing dataset using both
models. Second, the `confusionMatrix` command is used to get the
estimate on the out-of-sample error (i.e., 1 - accuracy). Note that
only the overall statistics is reported since the report is rather
large. Again, both models obtain accuracies above 90% but the random
forest performs best by a minor margin. Actually the error estimates
from the 10-fold cross-validations are very close to the ones obtained
in the probing dataset, suggesting that this partitioning may not have
been required to get an accurate out-of-sample error estimate and
train the model on more predictions in the first place.

For random forests:

```r
pred.rf.probing <- predict(model.rf, probing)
round(confusionMatrix(pred.rf.probing, probing$classe)$overall, digits=3)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##          0.991          0.988          0.988          0.993          0.284 
## AccuracyPValue  McnemarPValue 
##          0.000            NaN
```

For boosting:

```r
pred.boost.probing <- predict(model.boost, probing)
round(confusionMatrix(pred.boost.probing, probing$classe)$overall, digits=3)
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##          0.961          0.951          0.957          0.966          0.284 
## AccuracyPValue  McnemarPValue 
##          0.000          0.000
```


#### Predicting test cases

Instead of using model ensembling and stacking, let's simply perform
the predictions with both models and simply check whether both models
agree or not. If yes, the models seem to work very nicely and the
predictions of the test cases are more confident. Thus, the results
can be written to the files (uses the supplied code) and submitted to
the course website.


```r
pred.rf.testing <- predict(model.rf, testing)
pred.boost.testing <- predict(model.boost, testing)
cat("Do both models agree on all test cases?",
    all(pred.rf.testing == pred.boost.testing), "\n")
```

```
## Do both models agree on all test cases? TRUE
```

```r
answers <- pred.rf.testing

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
  
pml_write_files(answers)
```

By automated grading, all test cases turned out to correct.
