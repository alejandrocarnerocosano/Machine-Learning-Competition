source("settings.R")

### knn - Prediction function
# train: full imputed data set
# test: set with no missing values in feature values and only NA in target column
knn_regr_pred <- function(train, test, k, target = "song_popularity") {
  test[, target] <- NA
  data <- rbind(train, test)
  test_row_ids <- seq(nrow(train) + 1, nrow(train) + nrow(test))
  result <- kNN(data, k=k, variable = target, weightDist = FALSE)
  return(result[test_row_ids, ])
}

### MAPE calculation function (with shift + 1)
mape <- function(actual, predicted, epsilon = 1e-8) {
  100 * mean(abs((actual - predicted) / (actual + epsilon)))
}

### knn-tuning function
## input
# data: full imputed data set
# folds: folds of CV
# target: column name in the data set corresponding to the target variable
# search_space: values of k to try
knn_regr_tuning_k <- function(data, folds = 10, target = "song_popularity", search_space = NULL) {
  n <- nrow(data)
  
  if (is.null(search_space)) {
    k_values <- seq(sqrt(n) - 10, sqrt(n) + 10)
  } else {
    k_values <- search_space
  }
  
  fold_ids <- sample(rep(1:folds, length.out = n))
  
  results <- data.frame(k = k_values,
                        MAPE = NA,
                        RMSE = NA)
  
  for (i in seq_along(k_values)) {
    k <- k_values[i]
    fold_mape <- numeric(folds)
    fold_rmse <- numeric(folds)
    
    for (f in seq_len(folds)) {
      train <- data[fold_ids != f, ]
      test <- data[fold_ids == f, ]
      
      preds <- knn_regr_pred(train, test, k = k, target = target)[, target]
      
      fold_mape[f] <- mape(test[[target]], preds)
      fold_rmse[f] <- rmse(test[[target]], preds)
    }
    
    results$MAPE[i] <- mean(fold_mape)
    results$RMSE[i] <- mean(fold_rmse)
    message(sprintf("k = %d | CV-MAPE = %.3f | CV-RMSE = %.3f",
                    k, results$MAPE[i], results$RMSE[i]))
  }
  
  best_row <- results[which.min(results$MAPE), ]
  cat("Best K = ", best_row$k, "with average MAPE =", round(best_row$MAPE, 3))
  
  return(results)
}


##### knn: Model Tuning and Test #####
set.seed(123)
data_knn_vim <- read.csv(file.path(path_intermediate, "train_knn_vim.csv"))

# Try out different k
cv_results <- knn_regr_tuning_k(data_knn_vim, search_space = c(1, 10, 50, 100),
                                folds = 5, target = "song_popularity")


# plot(cv_results$k, cv_results$MAPE, type = "b",
#      main = "CV-MAPE for different k",
#      xlab = "k", ylab = "MAPE")

saveRDS(cv_results, file = "Results/knn_vim_tuning.rds")


