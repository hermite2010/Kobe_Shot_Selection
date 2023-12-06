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
Kobe_data_OG$shot_made_flag = as.factor(Kobe_data_OG$shot_made_flag)
Kobe_train <- na.omit(Kobe_data_OG)
Kobe_test <- Kobe_data_OG[apply(is.na(Kobe_data_OG), 1, any), ]

Kobe_train[sapply(Kobe_train, is.character)] <- lapply(Kobe_train[sapply(Kobe_train, is.character)], as.factor)
Kobe_test[sapply(Kobe_test, is.character)] <- lapply(Kobe_test[sapply(Kobe_test, is.character)], as.factor)

Kobe_train$shot_made_flag <- as.factor(Kobe_train$shot_made_flag)

Kobe_recipe <- recipe(shot_made_flag ~., data=Kobe_train) %>% 
  # step_mutate(game_event_id = as.factor(game_event_id),
  #             game_id = as.factor(game_id),
  #             period = as.factor(period),
  #             playoffs = as.factor(playoffs)) %>% 
  step_select(combined_shot_type,
              shot_distance,
              shot_type,
              shot_zone_basic,
              shot_made_flag) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(shot_made_flag))

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

# ## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=nb_folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

#stopCluster(cl)

## find best tuning parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize workflow and prediction 
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=Kobe_train)

nb_preds <- final_wf %>%
  predict(new_data = Kobe_test, type="prob") %>% 
  rename(shot_made_flag = .pred_1) %>% 
  select(shot_id,shot_made_flag)

## Write it out
vroom_write(x=nb_preds, file="NaiveKobe.csv", delim=",")