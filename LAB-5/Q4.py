# Importing necessary libraries
library(bnlearn)
library(e1071)
library(ggplot2)
# Initialize an empty vector to store accuracy results
accuracy_results <- numeric()
# Define custom color palette
custom_palette <- c("#FFEDA0", "#F03B20")  # Example custom colors

# Loop 20 times
for (i in 1:20) {
  # Split data into training set and testing set
  sample <- sample.int(n = nrow(data), size = floor(0.7 * nrow(data)), replace = FALSE)
  data.train <- data[sample, ]
  data.test <- data[-sample, ]
  
  # Build the naive Bayes classifier
  nb_grades <- naiveBayes(QP ~ ., data = data.train)
  
  # Predict using the classifier
  p <- predict(nb_grades, data.test)
  
  # Confusion matrix
  cm <- table(prediction = p, true = data.test$QP)
  
  # Calculate accuracy
  accuracy <- sum(diag(cm)) / sum(cm)
  
  # Store accuracy results
  accuracy_results <- c(accuracy_results, accuracy)
  
  # Visualize confusion matrix as heatmap
  cm_df <- as.data.frame(cm)
  plot_title <- paste("Confusion Matrix-Iteration", i)
  print(
    ggplot(data = cm_df, aes(x = prediction, y = true, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), vjust = 1) +
      scale_fill_gradient(low = custom_palette[1], high = custom_palette[2]) +  # Change colors here
      labs(title = plot_title, x = "Predicted Class", y = "True Class") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
}
# Get mean of accuracy results
print(mean(accuracy_results))