# Importing necessary library
library(bnlearn)

# Reading the data
data <- read.table("2020_bn_nb_data.txt", header = TRUE, col.names = c("EC100", "EC160", "IT101", "IT161", "MA101", "PH100", "PH160", "HS101", "QP"))

# Converting character variables to factor variables
data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)], as.factor)

# Learning the Bayesian network structure
bn_struct <- hc(data[,-9], score = 'k2')

# Fitting the data to the Bayesian network
fitted_bn <- bn.fit(bn_struct, data[,-9])

# Predicting the grade obtained in PH100 based on evidence provided
prediction_PH100 <- data.frame(cpdist(fitted_bn, nodes = c("PH100"),
                                      evidence = list(EC100 = "DD", IT101 = "CC", MA101 = "CD")))
my_table <- table(prediction_PH100)
my_table <- my_table / sum(my_table)
print(my_table)