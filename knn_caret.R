source("settings.R")

data_knn_caret <- read.csv(file.path(path_intermediate, "train_knn_caret.csv"))

# Prepare the dataset for the knn using FAMD ----------------------------------------------------------------------
X <- data_knn_caret %>%
  select(-song_popularity)
y <- data_knn_caret$song_popularity

res_famd <- FAMD(X, ncp = 10, graph = FALSE)
coords <- as.data.frame(res_famd$ind$coord)
coords$song_popularity <- y


# Train the knn Model ---------------------------------------------------------------------------------------------
set.seed(123)

control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
tuneGrid <- expand.grid(
  kmax = c(3, 7, 11, 21, 41, 81, 111),
  distance = 2,
  kernel = c("rectangular", "triangular", "gaussian")
)


knnModel <- train(
  song_popularity ~ .,
  data = coords,
  method = "kknn",
  metric = "RMSE",
  trControl = control,
  tuneGrid = tuneGrid
)


knnModel$bestTune
knnModel$results

plot(knnModel, xlab = "k", main = "Repeated 10-fold CV Tuning")

saveRDS(knnModel$results, file = "Results/knn_caret_tuning.rds")


# pred <- predict(knnModel, coords)
# rmse(coords$song_popularity, pred)
# mape(coords$song_popularity, pred)
