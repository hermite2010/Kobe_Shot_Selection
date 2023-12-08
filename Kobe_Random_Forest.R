library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
library(ggmosaic)  # For plotting
library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(discrim) # FOR NAIVE BAYES
library(kknn)
library(themis) # for smote

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

# Recipe to use for all models
Kobe_recipe <- recipe(shot_made_flag ~ ., data = Kobe_train) %>%
  update_role(shot_id, new_role = "ID") %>% 
  step_dummy(all_nominal_predictors())

# Classification Forest ---------------------------------------------------
# Set Up the Engine
class_for_mod <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Workflow and model and recipe
class_for_wf <- workflow() %>%
  add_recipe(Kobe_recipe) %>%
  add_model(class_for_mod)

## set up grid of tuning values

#class_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(Kobe_train)-1))),
#                                  min_n(),
#                                  levels = 6)

class_tuning_grid <- expand.grid(mtry = c(8,9,10),
                                 min_n = 40:45)

## set up k-fold CV
class_folds <- vfold_cv(Kobe_train, v = 4, repeats=1)

## Set up Parallel processing
num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

## Run the CV
CV_results <- class_for_wf %>%
  tune_grid(resamples=class_folds,
            grid=class_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

stopCluster(cl)
## find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and prediction 
final_wf <- class_for_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Kobe_train)

class_for_preds <- final_wf %>%
  predict(new_data = Kobe_test, type="prob") 

class_for_submit <- as.data.frame(cbind(Kobe_test$shot_id, class_for_preds$.pred_1)) %>% 
  rename("shot_id" = "V1", "shot_made_flag" = "V2")

## Write it out
vroom_write(x=class_for_preds, file="Kobe_Shot_Selection/ClassForest_2.csv", delim=",")