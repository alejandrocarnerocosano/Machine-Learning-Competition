source("settings.R")


# Functions -------------------------------------------------------------------------------------------------------
perform_famd_split <- function(train_data, test_data, target, ncp = 10) {
  # Split in X (Features) and y (target)
  X_train <- train_data[, !names(train_data) %in% target, drop = FALSE]
  y_train <- train_data[, target, drop = FALSE]
  
  X_test <- test_data[, !names(test_data) %in% target, drop = FALSE]
  y_test <- test_data[, target, drop = FALSE]
  
  # fit FAMD on training data
  famd_model <- FAMD(X_train, ncp = ncp, graph = FALSE)
  
  # coordinates of training data
  train_coords <- as.data.frame(famd_model$ind$coord)
  train_coords <- cbind(y_train, train_coords)
  names(train_coords)[1] <- target
  
  # coordinates of test data (via predict)
  test_coords <- as.data.frame(predict(famd_model, newdata = X_test)$coord)
  test_coords <- cbind(y_test, test_coords)
  names(test_coords)[1] <- target
  
  
  colnames(train_coords) <- make.names(colnames(train_coords))
  colnames(test_coords) <- make.names(colnames(test_coords))
  return(list(train = train_coords,
              test = test_coords))
}


knn_caret_test <- function(data, cv_test_folds, cv_tuning_folds,
                           target = "song_popularity", tuning_search_space_k = c(3, 9, 27, 81, 100, 111)) {
  target_vals <- data[, target]
  
  outer_folds <- createFolds(target_vals, k = cv_test_folds,
                             list = TRUE, returnTrain = TRUE)
  
  # Two empty data sets to store the results
  test_results <- data.frame(OuterFold = numeric(0),
                             k = numeric(0),
                             RMSE = numeric(0))
  
  overall_tuning_res <- data.frame(OuterFold = numeric(0),
                                   kmax = numeric(0),
                                   RMSE = numeric(0),
                                   Rsquared = numeric(0),
                                   MAE = numeric(0),
                                   RMSESD = numeric(0),
                                   RsquaredSD = numeric(0),
                                   MAESD = numeric(0))
  
  # Outer Loop for Testing
  for (i in seq_along(outer_folds)) {
    
    cat("Outer Fold:", i, "\n")
    
    # Train/ Test Split
    train_indices <- outer_folds[[i]]
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    
    # Prepare data for knn using FAMD
    famd_split <- perform_famd_split(train_data, test_data, target)
    train_famd <- famd_split$train
    test_famd <- famd_split$test
    
    # Inner Loop: cv_tuning_folds-CV for tuning
    inner_control <- trainControl(
      method = "cv",
      number = cv_tuning_folds
    )
    
    knn_grid <- expand.grid(kmax = tuning_search_space_k,
                            distance = 2,
                            kernel = "gaussian")
    
    formula <- as.formula(paste(target, "~ ."))
    knn_model <- train(
      formula,
      data = train_famd,
      method = "kknn",
      trControl = inner_control,
      tuneGrid = knn_grid,
      metric = "RMSE"
    )
    # --> already performs model tuning using specified CV 
    #     and afterwards learns model on whole trainset with best_k
    
    tuning_res <- knn_model$results
    # --> dataframe with tuning results (RMSE and other metrics for every k)
    
    best_k <- knn_model$bestTune$kmax
    
    cat("Tuning suggested k = ", best_k, "\n")
    
    # Save tuning results
    overall_tuning_res <- rbind(overall_tuning_res, cbind(OuterFold = i, tuning_res))
    
    # Test the tuned model using outer test set
    preds <- predict(knn_model, newdata = test_famd)
    
    # Performance estimation
    rmse <- RMSE(preds, test_data[, target])
    
    # Save test result
    test_results <- rbind(test_results,
                          data.frame(OuterFold = i,
                                     k = best_k,
                                     RMSE = rmse))
    
    cat("Test result using k = ", best_k, ": RMSE = ", round(rmse, 3), "\n")
  }
  
  summary_test_results <- test_results %>%
    summarise(mean_RMSE = mean(RMSE),
              sd_RMSE = sd(RMSE))
  
  return(list(
    tuning_results = overall_tuning_res,
    test_results = test_results,
    test_summary = summary_test_results
  ))
}


# Perform Test ----------------------------------------------------------------------------------------------------
set.seed(123)

data_knn_caret <- read.csv(file.path(path_intermediate, "train_knn_caret.csv"))

results <- knn_caret_test(data_knn_caret, cv_test_folds = 3, cv_tuning_folds = 10)

results$tuning_results
results$test_results
results$test_summary



# Save Test Results -----------------------------------------------------------------------------------------------

saveRDS(results$tuning_results, file = "Results/KNN/knn_caret_tuning.rds")
saveRDS(results$test_results, file = "Results/KNN/knn_caret_test.rds")
saveRDS(results$test_summary, file = "Results/KNN/knn_caret_test_summary.rds")

