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
Kobe_data_OG$shot_made_flag = as.factor(Kobe_data_OG$shot_made_flag)
Kobe_train <- na.omit(Kobe_data_OG)
Kobe_test <- Kobe_data_OG[apply(is.na(Kobe_data_OG), 1, any), ]

Kobe_train[sapply(Kobe_train, is.character)] <- lapply(Kobe_train[sapply(Kobe_train, is.character)], as.factor)
Kobe_test[sapply(Kobe_test, is.character)] <- lapply(Kobe_test[sapply(Kobe_test, is.character)], as.factor)

Kobe_train$shot_made_flag <- as.factor(Kobe_train$shot_made_flag)

Kobe_recipe <- recipe(shot_made_flag ~., data=Kobe_train) %>% 
  step_mutate(game_event_id = as.factor(game_event_id),
              game_id = as.factor(game_id),
              period = as.factor(period),
              playoffs = as.factor(playoffs)) %>% 
  step_select(-team_name,
              -game_date) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(shot_made_flag))

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
  predict(new_data = Kobe_test, type="prob") %>% 
  rename(shot_made_flag = .pred_1) %>% 
  select(shot_id,shot_made_flag)

## Write it out
vroom_write(x=pen_log_preds, file="PenLogisticKobe.csv", delim=",")