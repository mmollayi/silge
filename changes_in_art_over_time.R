library(tidyverse)
library(tidytext)
library(tidymodels)
library(textrecipes)

artwork <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-01-12/artwork.csv")

tate_df <- artwork %>%
    filter(year > 1750) %>%
    select(year, medium) %>%
    na.omit() %>%
    arrange(year)

tate_df %>%
    unnest_tokens(word, medium) %>%
    count(word, sort = TRUE)

set.seed(123)
art_split <- initial_split(tate_df, strata = year)

art_train <- training(art_split)
art_test <- testing(art_split)


set.seed(234)
art_folds <- vfold_cv(art_train, strata = year)
art_folds

