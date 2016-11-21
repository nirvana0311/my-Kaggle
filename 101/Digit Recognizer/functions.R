
#mse
mse <- function(actual, predict){
  (sum((predict - actual)^2))/length(actual)
}


# R-squared computation
R_squared <- function(actual, predict){
  mean_of_obs <- mean(actual)
  
  SS_tot <- sum((actual - mean_of_obs)^2)
  SS_reg <- sum((predict - mean_of_obs)^2)
  SS_res <- sum((actual - predict)^2)
  
  R_squared <- 1 - (SS_res/SS_tot)
  R_squared
}

R_squared1 <- function(actual, predict, mse){
  mean_of_obs <- mean(actual)
  
  SS_tot <- sum((actual - mean_of_obs)^2)
  SS_res <- mse*length(actual)
  
  R_squared <- 1 - (SS_res/SS_tot)
  R_squared
}

# remove columns with all NAs
rmBlankCols <- function(data){
  blankCols <- which(apply(data, 2, function(x)all(is.na(x))))
  data[, -blankCols]
}


# remove columns with all "none"
rmNoneCol <- function(data){
  NACols <- which(apply(data,2,function(x)all(x=="none")))
  if(!is.integer0(NACols)){
    data <- data[, -NACols]
  }    
  data
}

# replace empty cell with string "none"
replNone <- function(data){
  data=as.matrix(data)
  data[is.na(data)=="TRUE"]<- "none"          
  data[data==""] <- "none"
  data[data=="#####"] <- "none"
  as.data.frame(data)
}     


# convert RESULT column from binary to 0/1 
RESULT_binary_to_01 <- function(data, feature){
  data[, feature] <- as.character(data[, feature])
  data[which(data[, feature]=="PASS"), "RESULT"] <- 0
  data[which(data[, feature]=="FAIL"), "RESULT"] <- 1
  data[, feature] <-  as.integer(data[, feature])
  data
}


#####################################################################################

# get integer(0) and return (TRUE, FALSE)
is.integer0 <- function(x){
  is.integer(x) && length(x) == 0L
}

# convert all 'character' columns to 'factor' columns in a data frame
chrToFactor <- function(data){
  tmp <- unclass(data)
  as.data.frame(tmp)
}

# split a dataset into (train, test, valid) datasets(return formats is a list) 
split.data <- function(data, seed=set.seed(123), train=0, test=0, valid=0){
  # check 
  if( as.numeric(train+valid+test) > 1 ){
    return (cat("The sum of parameters is not equal to 1!"))
  }
  
  test_size <- floor(test * nrow(data))
  #set the seed to make partition reproductible
  seed
  test_ind <- sample(seq_len(nrow(data)), size = test_size )
  d_test <- data[test_ind, ]
  d_train <-  data[-test_ind, ]    
  d_valid <- NULL
  
  # if exists the validation dataset
  if(!(train+test== 1)){
    valid_size <- floor((valid/(train+test)) * nrow(d_train))
    seed
    valid_ind <- sample(seq_len(nrow(d_train)), size = valid_size )
    d_valid <- d_train[valid_ind, ]
    d_train <- d_train[-valid_ind, ]
  }
  
  list("train"=d_train, "test"=d_test, "valid"=d_valid)
}

# convert all category('factor' and 'chr') variables to 'dummy' variables in a data frame
categoryToDummy <- function(data){
  # 'chr' to 'factor'
  data <- chrToFactor(data) 
  
  class.list <- lapply(data, class)
  #class.list <- lapply(class.list, function(x) x <- x[!(x=="ordered")])# remove 'order' element of 'order-factor' type
  
  colNames <- names(class.list)
  
  
  colType <- unlist(class.list, use.names=T)
  
  factor.index <- unname(which(colType == "factor"))
  
  # means exist 'factor' variables
  if (is.integer0(factor.index) == F){
    
    # extract 'non-factor' variables
    nonfactor.df <- data[, -factor.index, drop = FALSE]
    
    # extract 'factor' variables
    factor.df <- data[, factor.index, drop = FALSE]
    
    # convert 'factor' variables to 'dummy' variables
    dummy.df <- as.data.frame(model.matrix(~., factor.df))
    dummy.df[, "(Intercept)"] <- NULL
    
    # Rename 'dummy' variables : 
    # since that model.matrix() will create new col.names by 
    # appending level to original col.names without any symbol, 
    # which is hard to tell original variables from levels. 
    
    a <- sapply(factor.df, function(x)levels(x), simplify=F)
    a <- lapply(a, `length<-`, max(lengths(a)))# adding NA to make all list elements equal length
    list.levels <- as.data.frame(a)
    list.levels <- list.levels[-1, ,drop=FALSE] # remove first level (since that model.matrix creates dummy matrix by ignoring the first level)
    new.dummy.names <- vector()
    for (i in 1:ncol(list.levels)){
      # append colnames with levels to create new colnames
      newName <- paste(colnames(list.levels)[i], list.levels[is.na(list.levels[,i])==F, i], 
                       sep=":")
      new.dummy.names <-  append(new.dummy.names, newName)
    }
    
    colnames(dummy.df) <- new.dummy.names
    
    # combind 'non-factor' variables and 'dummy' variables
    data <- cbind(nonfactor.df, dummy.df)
  } 
  data
}

##################################################################################


