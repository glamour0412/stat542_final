## install.packages("recommenderlab")

set.seed(1234)

library("recommenderlab")
data("MovieLense")

#data <- read.csv("D:/stat542/stat542_final-main/noisy_ground_truth_matrix/df_fill3.csv", header=TRUE)
data <- read.csv("D:/stat542/ml-1m.dat", sep = ' ', header=FALSE)
ratings <- as(data, "realRatingMatrix")

#MovieLense100 <- MovieLense[rowCounts(MovieLense) > 100, ]
#MovieLense100

#train <- MovieLense100[1:300]
#rec <- Recommender(train, method = "UBCF")
#rec

#pre <- predict(rec, MovieLense100[301:302], n = 5)
#pre

#as(pre, "list")

#新加的准备手撸的部分
#n <- nrow(data)
#indices <- sample(n, size = n/2, replace = FALSE)
#train <- data[indices, ]
#test <- data[-indices, ]
#train <- as(train, "realRatingMatrix")
#test <- as(test, "realRatingMatrix")

#algo <- Recommender(ratings, method="SVD")

#pred <- predict(algo, ratings)

#rmse <- RMSE(pred, ratings@data)
#print(paste("RMSE:", rmse))
#end of 新加的

#scheme <- evaluationScheme(MovieLense100, method = "cross-validation", k = 10, given = -5, goodRating = 4)
scheme <- evaluationScheme(ratings, method = "split", train = 0.5, given = 5)

algorithms <- list(`SVD` = list(name = "SVD", param = NULL), `SVDF` = list(name = "SVDF", param = NULL), `user-based CF` = list(name = "UBCF", param = list(nn = 3)), `item-based CF` = list(name = "IBCF", param = list(k = 100)), `ALS` = list(name = "ALS", param = NULL), `LIBMF` = list(name = "LIBMF", param = NULL))

results <- evaluate(scheme, algorithms, type = "ratings")

plot(results, annotate = 2, legend = "topleft")