# Libraries
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)

set.seed(123)

# Load data
datos_train <- read.csv(file.path(path_intermediate, "train_imputed.csv"))
test_data <- read.csv(file.path(path_intermediate, "test_imputed.csv"))

# Training Control (Cross Validation)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)


grid <- expand.grid(cp = seq(0.0001, 0.05, length.out = 20))

set.seed(123)
modelo_cart <- train(
  song_popularity ~ .,
  data = datos_train,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = grid,
  control = rpart.control(
    minsplit = 20,
    minbucket = 10,
    maxdepth = 10
  ),
  metric = "RMSE"
)

print(modelo_cart)
plot(modelo_cart)

# best cart model
best_cart <- modelo_cart$finalModel
rpart.plot(best_cart)

# predictions
preds_test <- predict(modelo_cart, newdata = test_data)

sample_submission_cart <- data.frame(id = 1:length(preds_test), song_popularity = preds_test)

write.csv(sample_submission_cart, file = file.path("Results", "sample_submission_cart.csv"),
          row.names = FALSE)
