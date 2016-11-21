### packages ###
#install.packages("drat", repos="https://cran.rstudio.com")
#drat:::addRepo("dmlc")
#install.packages("mxnet")
require(mxnet) # for deep learning(DNN, CNN)

### load working path and user-defined functions ###
source("functions.R")

### read train & test data ###
train <- read.csv('data/train.csv')
test <- read.csv('data/test.csv')

### preparation ###
train.x <- t(train[, -1]/255)
train.y <- train[, 1]
test.x <- t(test/255)

#=====================================================================#
#=========================== DNN =====================================#
#===========================0.977(2016/11/18)=========================#

# define my own evaluate function (R-squared)
my.eval.metric <- mx.metric.custom(
  name = "R-squared", 
  function(real, pred) {
    mean_of_obs <- mean(real)
    
    SS_tot <- sum((real - mean_of_obs)^2)
    SS_reg <- sum((predict - mean_of_obs)^2)
    SS_res <- sum((real - predict)^2)
    
    R_squared <- 1 - (SS_res/SS_tot)
    R_squared
  }
)



# 輸入層
data <- mx.symbol.Variable("data")

# 第一隱藏層: 500節點，狀態是Full-Connected
fc1 <- mx.symbol.FullyConnected(data, name="1-fc", num_hidden=500)
# 第一隱藏層的激發函數: Relu
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
# 這裡引入dropout的概念
drop1 <- mx.symbol.Dropout(data=act1, p=0.5)

# 第二隱藏層: 400節點，狀態是Full-Connected
fc2 <- mx.symbol.FullyConnected(drop1, name="2-fc", num_hidden=400)
# 第二隱藏層的激發函數: Relu
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
# 這裡引入dropout的概念
drop2 <- mx.symbol.Dropout(data=act2, p=0.5)

# 輸出層：因為預測數字為0~9共十個，節點為10
output <- mx.symbol.FullyConnected(drop2, name="output", num_hidden=10)
# Loss Function: Softmax
dnn <- mx.symbol.SoftmaxOutput(output, name="dnn")

arguments(dnn)


mx.set.seed(0) 

# 訓練剛剛創造/設計的模型
dnn.model <- mx.model.FeedForward.create(
  dnn,       # 剛剛設計的DNN模型
  X=train.x,  # train.x
  y=train.y,  #  train.y
  ctx=mx.cpu(),  # 可以決定使用cpu或gpu
  num.round=10,  # iteration round
  array.batch.size=100, # batch size
  learning.rate=0.07,   # learn rate
  momentum=0.9,         # momentum  
  eval.metric=mx.metric.accuracy, # 評估預測結果的基準函式*
  initializer=mx.init.uniform(0.07), # 初始化參數
  epoch.end.callback=mx.callback.log.train.metric(100)
)

graph.viz(dnn.model$symbol$as.json())
# test prediction #
test.y <- predict(dnn.model, test.x)
test.y <- t(test.y)
test.y.label <- max.col(test.y) - 1

### write out the result ###
result <- data.frame(ImageId = 1:length(test.y.label),
                     label = test.y.label)

write.csv(result,"result/20161120_mxnet_dnn.csv", row.names = F)




#=========================== CNN(LeNet) ==============================#
#===========================0.98600(2016-11-18========================#
#=====================================================================#
#=====================================================================#
#=====================================================================#
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28, 28, 1, ncol(test.x))

# 輸入層
data <- mx.symbol.Variable('data')

# 第一卷積層，windows的視窗大小是 5x5
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20, name="1-conv")
# 第一卷積層的激發函數：Relu
conv.act1 <- mx.symbol.Activation(data=conv1, act_type="relu", name="1-conv.act")
# 第一卷積層後的池化層，max，大小縮為 2x2
pool1 <- mx.symbol.Pooling(data=conv.act1, pool_type="max", name="1-conv.pool",
                           kernel=c(2,2), stride=c(2,2))
# 這裡引入dropout
drop1 <- mx.symbol.Dropout(data=pool1, p=0.5)

# 2-conv
conv2 <- mx.symbol.Convolution(data=drop1, kernel=c(5,5), num_filter=50, name="2-conv")
conv.act2 <- mx.symbol.Activation(data=conv2, act_type="relu", name="2-conv.act")
pool2 <- mx.symbol.Pooling(data=conv.act2, pool_type="max", name="2-conv.pool",
                           kernel=c(2,2), stride=c(2,2))
drop2 <- mx.symbol.Dropout(data=pool2, p=0.5)


flatten <- mx.symbol.Flatten(data=drop2)
# 再來，建立一個Full-Connected的隱藏層，500節點
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500, name="1-fc")
# 隱藏層的激發函式：Relu
fc.act1 <- mx.symbol.Activation(data=fc1, act_type="relu", name="1-fc.act")

# 輸出層：因為預測數字為0~9共十個，節點為10
output <- mx.symbol.FullyConnected(data=fc.act1, num_hidden=10, name="output")
# Loss Function: Softmax 
cnn <- mx.symbol.SoftmaxOutput(data=output, name="cnn")



mx.set.seed(0)
cnn.model <- mx.model.FeedForward.create(
  cnn,       # 剛剛設計的DNN模型
  X=train.array,  # train.x
  y=train.y,  #  train.y
  ctx=mx.cpu(),  # 可以決定使用cpu或gpu
  num.round=10,  # iteration round
  array.batch.size=100, # batch size
  learning.rate=0.07,   # learn rate
  momentum=0.7,         # momentum  
  eval.metric=mx.metric.accuracy, # 評估預測結果的基準函式*
  initializer=mx.init.uniform(0.05), # 初始化參數
  #wd=0.0001,
  epoch.end.callback=mx.callback.log.train.metric(100)
)

# test prediction #
test.y <- predict(cnn.model, test.array)
test.y <- t(test.y)
test.y.label <- max.col(test.y) - 1

### write out the result
result <- data.frame(ImageId = 1:length(test.y.label),
                     label = test.y.label)

write.csv(result, "result/20161121_mxnet_cnn.csv", row.names = F)






