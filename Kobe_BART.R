library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
#library(ggmosaic)  # For plotting
#library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(discrim) # FOR NAIVE BAYES
library(kknn)
library(parsnip)    # FOR BART
library(dbarts)
library(themis)

# Reading in the Data
Kobe_data_OG <- vroom("Kobe_Shot_Selection/data.csv") 
samp <- vroom("Kobe_Shot_Selection/sample_submission.csv")

Kobe_train <- na.omit(Kobe_data_OG)
Kobe_test <- Kobe_data_OG[apply(is.na(Kobe_data_OG), 1, any), ]

Kobe_train[sapply(Kobe_train, is.character)] <- lapply(Kobe_train[sapply(Kobe_train, is.character)], as.factor)
Kobe_test[sapply(Kobe_test, is.character)] <- lapply(Kobe_test[sapply(Kobe_test, is.character)], as.factor)

test.id <- Kobe_test$shot_id
Kobe_train$shot_id <- NULL
Kobe_test$shot_id <- NULL


train_y <- Kobe_train$shot_made_flag

Kobe_recipe <- recipe(shot_made_flag ~., data=Kobe_train) %>% 
  step_mutate(time_remaining = minutes_remaining*60+seconds_remaining) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(shot_made_flag)) %>% 
  step_normalize(all_predictors()) %>% 
  step_select(action_type,
              combined_shot_type,
              loc_x,
              loc_y,
              minutes_remaining,
              period,
              playoffs,
              season,
              shot_distance,
              shot_type,
              shot_zone_area,
              shot_zone_basic,
              shot_zone_range,
              game_date,
              matchup,
              opponent,
              time_remaining) %>% 
  step_pca(all_predictors(), threshold = 0.95)

## Set up Parallel processing
num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
if (num_cores > 4)
  num_cores = 10
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

prepped_recipe <- prep(Kobe_recipe)
baked_data <- bake(prepped_recipe, new_data=Kobe_train)

# Set the model
bart_mod <- parsnip::bart(mode = "classification",
                          engine = "dbarts",
                          trees = 25)

# Set workflow
bart_wf <- workflow() %>%
  add_recipe(Kobe_recipe) %>%
  add_model(bart_mod) %>% 
  fit(data = Kobe_train)

stopCluster(cl)

# Finalize and Predict 
bart_preds <- bart_wf %>%
  fit(data = Kobe_train) %>% 
  predict(new_data = Kobe_test, type="prob") %>% 
  rename(shot_made_flag = .pred_1) %>% 
  select(shot_id,shot_made_flag)


vroom_write(x=bart_preds, file="Kobe_BART.csv", delim=",") #"Amazon_AEAC_Kaggle/NaiveBayes.csv"
