source("settings.R")

impute_mice <- function(df) {
  imp_obj <- mice(df, method = "pmm", m = 5)
  df_imputed <- complete(imp_obj)
  return(df_imputed)
}

# imputed train.csv
train_raw <- read.csv(file.path(path_raw, "train.csv"))
train_imputed <- impute_mice(train_raw)

write.csv(train_imputed, file = file.path(path_intermediate, "train_imputed.csv"),
          row.names = FALSE)

# impute test.csv
test_raw <- read.csv(file.path(path_raw, "test.csv"))
test_imputed <- impute_mice(test_raw)

write.csv(test_imputed, file = file.path(path_intermediate, "test_imputed.csv"),
          row.names = FALSE)
