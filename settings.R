#########################################
##-----Setup file for the project-----##
########################################

# Load packages
packages <- c(
  "arules",
  "arulesViz",
  "caret",
  "dplyr",
  "e1071",
  "FactoMineR",
  "fastDummies",
  "ggcorrplot",
  "ggplot2",
  "ggpubr",
  "glue",
  "haven",
  "inspectdf",
  "lifecycle",
  "kernlab",
  "MASS",
  "Metrics",
  "mice",
  "mlbench",
  "naniar",
  "patchwork",
  "PerformanceAnalytics",
  "psych",
  "readr",
  "Rlof",
  "skimr",
  "solitude",
  "tidymodels",
  "tidyr",
  "tree",
  "VIM"
)


for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Data Paths
path_raw <- "Data/raw"
path_intermediate <- "Data/intermediate"

# Theme
theme_set(theme_minimal())

