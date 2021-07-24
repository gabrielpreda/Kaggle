data = read.csv("iris.csv")

saveRDS(data, "iris.rds")

r_data = readRDS("iris.rds")
library("arrow")

df <- read_parquet("iris.parquet")