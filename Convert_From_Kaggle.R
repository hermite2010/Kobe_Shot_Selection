library(tidyverse)
library(vroom)
# Read in ClassForest_Kobe_1.csv
kobe <- vroom("Kobe_Shot_Selection/ClassForest_Kobe_1.csv")

# Reading in the Data
Kobe_data_OG <- vroom("Kobe_Shot_Selection/data.csv") 
samp <- vroom("Kobe_Shot_Selection/sample_submission.csv")

# Turning all string variable into factors
Kobe_data_OG[sapply(Kobe_data_OG, is.character)] <- lapply(Kobe_data_OG[sapply(Kobe_data_OG, is.character)], as.factor)

# Creating New variables or Changing Variables
Kobe_data_OG$home <- as.numeric(grepl("vs.", Kobe_data_OG$matchup, fixed = TRUE))
Kobe_data_OG$away <- as.numeric(grepl("@", Kobe_data_OG$matchup, fixed = TRUE))
Kobe_data_OG$time_remaining <- Kobe_data_OG$minutes_remaining*60+Kobe_data_OG$seconds_remaining
Kobe_data_OG$lastminutes <- ifelse(Kobe_data_OG$time_remaining <= 180, 1, 0)

## Variables that I want to remove
# game_event_id, game_id
Kobe_data_OG <- Kobe_data_OG %>% select(-c(game_event_id, game_id))
# lat, lon, loc_x, loc_y
Kobe_data_OG <- Kobe_data_OG %>% select(-c(loc_x,loc_y,lat,lon))
# team_id, team_name
Kobe_data_OG <- Kobe_data_OG %>% select(-c(team_id, team_name))
# minutes_remaining, seconds_remaining
Kobe_data_OG <- Kobe_data_OG %>% select(-c(minutes_remaining, seconds_remaining))
# game_date
Kobe_data_OG <- Kobe_data_OG %>% select(-game_date)
# matchup
Kobe_data_OG <- Kobe_data_OG %>% select(-matchup)

# Select the variabels that I want to use to predict and Split the data
Kobe_train<- subset(Kobe_data_OG, !is.na(shot_made_flag))
Kobe_train$shot_made_flag <- as.factor(Kobe_train$shot_made_flag)
Kobe_test <- subset(Kobe_data_OG, is.na(shot_made_flag)) %>% 
  select(-shot_made_flag)

class_for_submit <- as.data.frame(cbind(Kobe_test$shot_id, kobe$.pred_1)) %>% 
  rename("shot_id" = "V1", "shot_made_flag" = "V2")

## Write it out
vroom_write(x=class_for_submit, file="Kobe_Shot_Selection/ClassForest.csv", delim=",")
