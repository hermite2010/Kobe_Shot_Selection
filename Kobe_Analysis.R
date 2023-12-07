library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
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
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(shot_made_flag))

# Naive Bayes -------------------------------------------------------------
## Set Up Model
nb_model <- naive_Bayes(Laplace=tune(), 
                        smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

## Workflow and model and recipe
nb_wf <- workflow() %>%
  add_recipe(Kobe_recipe) %>%
  add_model(nb_model)

## set up grid of tuning values
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 10)

## set up k-fold CV
nb_folds <- vfold_cv(Kobe_train, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=nb_folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and prediction 
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Kobe_train)

nb_preds <- final_wf %>%
  predict(new_data = Kobe_test, type="prob") 

nb_submit <- as.data.frame(cbind(Kobe_test$shot_id, nb_preds$.pred_1)) %>% 
  rename("shot_id" = "V1", "shot_made_flag" = "V2")


## Write it out
vroom_write(x=nb_submit, file="Kobe_Shot_Selection/NaiveKobe.csv", delim=",")


# BART --------------------------------------------------------------------

# Set the model
bart_mod <- parsnip::bart(mode = "classification",
                          engine = "dbarts",
                          trees = 25)

# Set workflow
bart_wf <- workflow() %>%
  add_recipe(Kobe_recipe) %>%
  add_model(bart_mod) %>% 
  fit(data = Kobe_train)

# Finalize and Predict 
bart_preds <- bart_wf %>%
  fit(data = Kobe_train) %>% 
  predict(new_data = Kobe_test, type="prob")

bart_submit <- as.data.frame(cbind(Kobe_test$shot_id, bart_preds$.pred_1)) %>% 
  rename("shot_id" = "V1", "shot_made_flag" = "V2")


vroom_write(x=bart_submit, file="Kobe_Shot_Selection/Kobe_BART.csv", delim=",")

# Penalized Logistic Regression -------------------------------------------
#Type of model
pen_log_model <- logistic_reg(mixture=tune(),
                              penalty=tune()) %>% 
  set_engine("glmnet")
# Set the Workflow
pen_log_workflow <- workflow() %>%
  add_recipe(Kobe_recipe) %>%
  add_model(pen_log_model)

## Grid of values to tune over
pen_log_tuning_grid <- grid_regular(penalty(),
                                    mixture(), # Always bewteen 0 and 1
                                    levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(Kobe_train, v = 3, repeats=1)

## Run the CV
CV_results <- pen_log_workflow %>%
  tune_grid(resamples=folds,
            grid=pen_log_tuning_grid,
            metrics=metric_set(roc_auc)) 

## Find Best Tuning Parameters
pen_log_bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_wf <- pen_log_workflow %>%
  finalize_workflow(pen_log_bestTune) %>%
  fit(data=Kobe_train)

## Predict
pen_log_preds <- final_wf %>%
  predict(new_data = Kobe_test, type="prob")

pen_log_submit <- as.data.frame(cbind(Kobe_test$shot_id, pen_log_preds$.pred_1)) %>% 
  rename("shot_id" = "V1", "shot_made_flag" = "V2")

## Write it out
vroom_write(x=pen_log_submit, file="Kobe_Shot_Selection/PenLogisticKobe.csv", delim=",")
