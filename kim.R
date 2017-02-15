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
         Region = gsub('[[:digit:]]','',x=TourneySeeds$Seed))
#去除原先的seed
TourneySeeds <- TourneySeeds %>% select(-Seed)

#按sample_submission排序,input 数据

game_to_predict <- cbind(sample_submission$id,
                         colsplit(sample_submission$id,
                                  "_",
                                  names = c('season','team1','team2'))
)
#训练集

#合并常规赛和竞标赛数据 (compact)

temp <- left_join(game_to_predict,
                  TourneySeeds, 
                  by=c("season"="Season", "team1"="Team"))

games_to_predict <- left_join(temp, 
                              TourneySeeds, 
                              by=c("season"="Season", "team2"="Team"))

colnames(games_to_predict)[c(1,5:8)] <- c("id", "team1seed",'team1region',"team2seed",'team2region')

#join compact result

compact_result <- rbind(RegularSeasonCompactResults,TourneyCompactResults)

compact_result <- compact_result[order(compact_result$Season,compact_result$Daynum),]

#join detailed result

detailed_result <- rbind(RegularSeasonDetailedResults,TourneyDetailedResults)
detailed_result <- detailed_result[order(detailed_result$Season,detailed_result$Daynum),]




#left_join(as.data.frame(TourneyCompactResults), 
#          TourneySeeds, 
#          by=c("Season", "Wteam"="Team"))


