# Importing necessary libraries
library(bnlearn)
library(bnclassify)
library(ggplot2)

# Reading the data
data <- read.table("data.txt", header = TRUE, col.names = c("EC100", "EC160", "IT101", "IT161", "MA101", "PH100", "PH160", "HS101", "QP"))

# Convert character variables to factor variables
data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)], as.factor)

# Set seed values
set.seed(0)

# Initialize an empty vector to store accuracy results
accuracy_results <- numeric()

# Initialize an empty list to store confusion matrices
confusion_matrices <- list()

# Loop 20 times
for (i in 1:20) {
  train_indices <- sample(nrow(data), 0.7 * nrow(data)) # 70% train, 30% test
  data.train <- data[train_indices, ]
  data.test <- data[-train_indices, ]
  
  # Build the TAN classifier on the training data using the tan_cl function from the bnlearn package.
  tn <- tan_cl("QP", data.train)
  
  # Fit the TAN classifier to the training data using the lp function
  tn <- lp(tn, data.train, smooth = 1)
  
  # Use the predict function to predict the grades of the test data.
  p <- predict(tn, data.test)
  
  # Compute the confusion matrix using the table function
  cm <- table(predicted = p, true = data.test$QP)
  confusion_matrices[[i]] <- cm
  
  # Compute the accuracy of the prediction using the accuracy function from the bnclassify package
  accuracy <- bnclassify:::accuracy(p, data.test$QP)
  
  # Store the accuracy in the vector
  accuracy_results <- c(accuracy_results, accuracy)
}

# Getting mean of accuracy results
mean_accuracy <- mean(accuracy_results)
print(paste("Mean Accuracy:", mean_accuracy))

# Define custom color palette
custom_palette <- c("lightblue", "darkblue", "lightgreen", "darkgreen")

# Plotting confusion matrix heatmaps
for (i in 1:length(confusion_matrices)) {
  cm <- confusion_matrices[[i]]
  cm_df <- as.data.frame(cm)
  plot_title <- paste("Confusion Matrix-Iteration", i)
  print(
    ggplot(data = cm_df, aes(x = predicted, y = true, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), vjust = 1) +
      scale_fill_gradient(low = custom_palette[1], high = custom_palette[2]) +  # Change colors here
      labs(title = plot_title, x = "Predicted Class", y = "True Class") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
}