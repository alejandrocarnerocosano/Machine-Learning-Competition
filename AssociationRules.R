library(arules)
library(arulesViz)
library(FactoMineR)
library(tidyverse)
source("settings.R")

#################################################
# Using pre-processed data

train <- read.csv(file.path(path_intermediate, "train_knn_caret.csv"))
train_orig <- read.csv(file.path(path_intermediate, "train_imputed.csv"))

train$song_popularity <- train_orig$song_popularity

#############################

# Discretize the data before using Apriori function

train <- train[,-1] #remove ID
train_discretized <- train

# variables already categorical
categorical_cols <- c("key", "audio_mode", "time_signature", "es_instrumental", "valence_binned")

#numerical columns to discretise with "frequency" criterium (same number of element in each interval)
freq_cols <- c("liveness", "acousticness", "speechiness", "song_duration_ms")

#numerical columns to discretise with "intervals" (the intervals has the same length)
interval_cols <- c("loudness", "danceability", "energy", "tempo", "song_popularity")

train_discretized[freq_cols] <- lapply(train[freq_cols], function(col) {
  discretize(col, method = "frequency", breaks = 3, labels = c("Low", "Medium", "High"))
})

train_discretized[interval_cols] <- lapply(train[interval_cols], function(col) {
  discretize(col, method = "interval", breaks = 3, labels = c("Low", "Medium", "High"))
})

train_discretized[categorical_cols] <- lapply(train[categorical_cols], as.factor)

#################################

# Transorm the db in transactional data and find the Item Frequency

train_discretized <- train_discretized[-6] #remove redundant column
train_transactions <- as(train_discretized, "transactions")
itemFrequencyPlot(train_transactions, topN = 10, type = "relative")

# looking for items containing information about song_popularity (target variable)
# check what is their support in order to decide an appropriate min_support

all_items <- itemLabels(train_transactions)

popularity_items <- grep("song_popularity=", all_items, value = TRUE)
popularity_transactions <- train_transactions[ , popularity_items]
popularity_freqs <- itemFrequency(popularity_transactions, type = "relative")

print("Frecuencias Relativas (Proporciones) para Popularidad:")
print(popularity_freqs)

#######################################

# Find the rules 
# The min_support is chosen based on the previous computation
# The confidence is chosen low because some rules with song_popularity in the rhs have low confidence

rules <- apriori(train_transactions, 
                 parameter = list(supp = 0.01, conf = 0.4, maxlen = 4))

# inspect the strongest rules
rules_filtrado <- arules::subset(rules, subset = confidence > 0.7 & support > 0.1 & lift > 1.1)
rules_sorted <- sort(rules_filtrado, by = "lift", decreasing = TRUE)
inspect(head(rules_sorted, 5))


#inspect only rules that has song popularity in the rhs

rules_popularity <- arules::subset(rules,
                                   subset = rhs %in% c("song_popularity=High", "song_popularity=Medium",
                                                       "song_popularity=Low") & lift > 1.1)
rules_sorted <- sort(rules_popularity, by = "confidence", decreasing = TRUE)
inspect(rules_sorted[1:5])


rules_filtrado_high <- arules::subset(rules,
                                      subset = rhs %in% "song_popularity=High" & lift > 1)
rules_filtrado_high <- sort(rules_filtrado_high, by = "support", decreasing = TRUE)
inspect(rules_filtrado_high[1:5])


rules_filtrado_low <- arules::subset(rules,
                                     subset = rhs %in% "song_popularity=Low" & lift > 1)
rules_filtrado_low <- sort(rules_filtrado_low, by = "support", decreasing = TRUE)
inspect(rules_filtrado_low[1:5])

