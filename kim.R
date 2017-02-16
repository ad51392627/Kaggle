#Kaggle, tidy data

rm(list = ls())
library(rpart)
library(readr)
library(knitr)
library(rpart.plot)
library(dplyr)
library(randomForest)
library(party)
library(caret)
library(cluster)
library(purrr)
library(tidyr)
library(e1071)
library(modelr)
library(ROSE)
library(nnet)
library(ROCR)
library(reshape2)
data_path <- "~/Github/Kaggle/Data"
RegularSeasonCompactResults <- read.csv(file.path(data_path,'RegularSeasonCompactResults.csv'))
RegularSeasonDetailedResults <- read.csv(file.path(data_path,'RegularSeasonDetailedResults.csv'))
Seasons <- read.csv(file.path(data_path,'Seasons.csv'))
TourneyCompactResults <- read.csv(file.path(data_path,'TourneyCompactResults.csv'))
TourneyDetailedResults <- read.csv(file.path(data_path,'TourneyDetailedResults.csv'))
TourneySeeds <- read.csv(file.path(data_path,'TourneySeeds.csv'))
TourneySlots <- read.csv(file.path(data_path,'TourneySlots.csv'))
sample_submission <- read.csv(file.path(data_path,'sample_submission.csv'))
#提取seed数据，分开成seednum 和 region
TourneySeeds <- 
  TourneySeeds %>%
  mutate(SeedNum = gsub('[^[:digit:]]','',x=Seed),
         Region = gsub('[[:digit:]]','',x=Seed))
#去除原先的seed
TourneySeeds <- TourneySeeds %>% select(-Seed)

#按sample_submission排序,input 数据

game_to_predict <- cbind(sample_submission$id,
                         colsplit(sample_submission$id,
                                  "_",
                                  names = c('season','team1','team2')))


temp <- left_join(game_to_predict,
                  TourneySeeds, 
                  by=c("season"="Season", "team1"="Team"))

game_to_predict <- left_join(temp, 
                              TourneySeeds, 
                              by=c("season"="Season", "team2"="Team"))

colnames(game_to_predict)[c(1,5:8)] <- c("id", "team1seed",'team1region',"team2seed",'team2region')

#去除region

game_to_predict <- game_to_predict %>% select(-team1region,-team2region)

game_to_predict <- game_to_predict %>%
  mutate(SeedDiff = as.integer(team1seed) - as.integer(team2seed))

#训练集

#锦标赛数据（seed Diff模型）
training_data = left_join(TourneyCompactResults,
                          TourneySeeds,
                          by = c("Season" = "Season", "Wteam" = "Team"))

training_data = left_join(training_data,
                          TourneySeeds,
                          by = c("Season" = "Season", "Lteam" = "Team"))

training_data <- 
  training_data %>% 
                mutate(SeedDiff = as.integer(SeedNum.x) - as.integer(SeedNum.y))

training_data <- training_data %>% select(-SeedNum.x,-SeedNum.y,-Region.x,-Region.y,-Wloc,Numot,-Daynum,-Wscore,-Lscore,-Numot)

colnames(training_data)[c(1:3)] <- c("season", "team1",'team2')

training_set_1 <- training_data %>% mutate(result = as.factor(1))
training_set_2 <- training_data %>% mutate(SeedDiff = -SeedDiff,
                                           result = as.factor(0))

training_data <- rbind(training_set_1,training_set_2)

training_data <- training_data %>% select(-season,-team1,-team2)

#split test data and training data

splits <- createDataPartition(training_data$result, p=0.84) #80% train, 50% test
train <- training_data[splits$Resample1,]
test <- training_data[-splits$Resample1,]

#锦标赛数据







#modelling using h2o deeplearning

library(h2o)
localh2o <- h2o.init(min_mem_size = "512m", max_mem_size = "12g")

train_h2o <- as.h2o(train, destination_frame = "train")
test_h2o <- as.h2o(test, destination_frame = "test")

h2o_grid_search <- h2o.grid("deeplearning", 
                            grid_id = 'h2orm',
                            y = which((names(train_h2o) == "result")),
                            x = which(!(names(train_h2o) == "result")),
                            training_frame = train_h2o,
                            nfolds = 10,
                            score_each_iteration = T,
                            overwrite_with_best_model = T,
                            standardize = T,
                            activation = 'RectifierWithDropout',
                            epochs = 50,
                            stopping_rounds = 5,
                            stopping_metric = 'logloss',
                            stopping_tolerance = 1e-3,
                            shuffle_training_data = T,
                            hyper_params = list(hidden = c(c(200,200),c(64,64,64),c(512),c(32,32,32,32,32)),
                                                input_dropout_ratio = c(0.2,0.5,0.7), 
                                                l1 = c(1e-4,1e-3),
                                                l2 = c(1e-4,1e-3))
)

grid <- h2o.getGrid('h2orm', sort_by = 'logloss', decreasing = FALSE)

grid@summary_table[1,]

best_model <- h2o.getModel(grid@model_ids[[1]])
best_model

model_1 <- h2o.deeplearning(
  model_id = 'no_1',
  y = which((names(train_h2o) == "result")),
  x = which(!(names(train_h2o) == "result")),
  training_frame = train_h2o,
  keep_cross_validation_predictions = T,
  nfolds = 10,
  score_each_iteration = T,
  overwrite_with_best_model = T,
  standardize = T,
  activation = 'RectifierWithDropout',
  epochs = 30,
  stopping_rounds = 5,
  stopping_metric = 'logloss',
  stopping_tolerance = 1e-3,
  shuffle_training_data = T,
  hidden = c(200,200,200),
  input_dropout_ratio = 0.5,
  hidden_dropout_ratios = c(0.5,0.5,0.3),
  l1 = 1e-3,
  l2 = 1e-3)

model_2 <- h2o.deeplearning(
  model_id = 'no_2',
  y = which((names(train_h2o) == "result")),
  x = which(!(names(train_h2o) == "result")),
  training_frame = train_h2o,
  keep_cross_validation_predictions = T,
  nfolds = 10,
  score_each_iteration = T,
  overwrite_with_best_model = T,
  standardize = T,
  activation = 'TanhWithDropout',
  epochs = 30,
  stopping_rounds = 5,
  stopping_metric = 'logloss',
  stopping_tolerance = 1e-3,
  shuffle_training_data = T,
  hidden = c(64,64,64),
  input_dropout_ratio = 0.5, 
  hidden_dropout_ratios = c(0.5,0.3,0.3),
  l1 = 1e-3,
  l2 = 1e-3)

model_3 <- h2o.deeplearning(
  model_id = 'no_3',
  y = which((names(train_h2o) == "result")),
  x = which(!(names(train_h2o) == "result")),
  training_frame = train_h2o,
  keep_cross_validation_predictions = T,
  nfolds = 10,
  score_each_iteration = T,
  overwrite_with_best_model = T,
  standardize = T,
  activation = 'RectifierWithDropout',
  epochs = 25,
  stopping_rounds = 5,
  stopping_metric = 'logloss',
  stopping_tolerance = 1e-3,
  shuffle_training_data = T,
  hidden = c(32,32,32,32,32),
  input_dropout_ratio = 0.5, 
  hidden_dropout_ratios = c(0.5,0.5,0.5,0.5,0.5),
  l1 = 1e-2,
  l2 = 1e-2)

perf_model_1 <- h2o.performance(model_1, test_h2o)

#random forest

model_2 <- h2o.randomForest(y = which((names(train_h2o) == "result")),
                            x = which(!(names(train_h2o) == "result")),
                            training_frame = train_h2o,
                            nfolds = 10,
                            ntrees = 1000,
                            max_depth = 20,
                            mtries = 1)


perf_model_2 <- h2o.performance(model_2, test_h2o)
perf_model_3 <- h2o.performance(model_3, test_h2o)

pred = h2o.predict(model_1,test_h2o, type = 'prob')

table(bank_data_h2o_test$default,pred)



#合并常规赛和竞标赛数据 (compact)


#join compact result

compact_result <- rbind(RegularSeasonCompactResults,TourneyCompactResults)

compact_result <- compact_result[order(compact_result$Season,compact_result$Daynum),]

#join detailed result

detailed_result <- rbind(RegularSeasonDetailedResults,TourneyDetailedResults)
detailed_result <- detailed_result[order(detailed_result$Season,detailed_result$Daynum),]




#left_join(as.data.frame(TourneyCompactResults), 
#          TourneySeeds, 
#          by=c("Season", "Wteam"="Team"))


