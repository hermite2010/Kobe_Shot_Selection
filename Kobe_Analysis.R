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
Kobe_data_OG <- vroom("Kobe_Shot_Selection/data.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
samp <- vroom('Kobe_Shot_Selection/sample_submission.csv')
Kobe_data_OG$shot_made_flag = as.factor(Kobe_data_OG$shot_made_flag)
Kobe_train <- na.omit(Kobe_data_OG)
Kobe_test <- Kobe_data_OG[apply(is.na(Kobe_data_OG), 1, any), ]



Kobe_recipe <- recipe(shot_made_flag ~., data=Kobe_train) %>% 
  step_mutate(game_event_id = as.factor(game_event_id)) %>% 
  step_mutate(game_id = as.factor(game_id)) %>% 
  step_mutate(period = as.factor(period)) %>% 
  step_mutate(playoffs = as.factor(playoffs)) #%>% 
#  step_other(all_nominal_predictors(), threshold = .001) %>%
#  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION)) %>% 
#  step_smote(all_outcomes(), neighbors = 5) #%>% 
#  step_upsample()
# OR step_downsample()

# apply the recipe to your data
prepped_recipe <- prep(Kobe_recipe)
baked_data <- bake(prepped_recipe, new_data=Kobe_train)

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
class_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(Kobe_train)-1))),
                                  min_n(),
                                  levels = 6)

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
  predict(new_data = Kobe_test, type="prob") %>% 
  rename(shot_made_flag = .pred_1) %>% 
  select(shot_id,shot_made_flag)

## Write it out
vroom_write(x=class_for_preds, file="ClassForest_SMOTE.csv", delim=",")