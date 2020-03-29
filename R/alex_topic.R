library(tidyverse)
library(topicmodels)
library(tidytext)
library(SnowballC)
# install.packages('LDAvis')
library(LDAvis)
# install.packages('textstem')
library(textstem)
library(scales)
library(parallel)
library(doParallel)

a <- read_csv('./a_reviews.csv') %>%
  select(-c(X1))

# s <- read_csv('./s_reviews.csv') %>%
  # select(-c(X1))

# wa <- read_csv('../wa_reviews.csv') %>%
  # select(-c(X1))

# we <- read_csv('../full_reviews.rds') %>%
  # select(-c(X1))

animation_tidy <- a %>%
  select(V1, V2) %>%
  unnest_tokens(word, V2) %>%
  anti_join(stop_words)

animation_tidy <- animation_tidy %>%
  mutate(lemma = lemmatize_words(word)) 

wordCount <- animation_tidy %>%
  group_by(word) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

wordCount <- animation_tidy %>%
  group_by(lemma) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  filter(count > 20)
  
final_words <- animation_tidy %>%
  filter(lemma %in% wordCount$lemma) %>%
  select(V1, lemma)

dtm_a <- animation_tidy %>%
  count(V1, lemma) %>%
  cast_dtm(V1, lemma, n)

## LDA model
LDA_animation <- LDA(dtm_a, k = 15, method = 'Gibbs',
                     control = list(alpha = 1/10, iter=50, 
                                    burnin=10000, seed = 1234))

  
topics <- tidy(LDA_animation, matrix = 'beta')

top_topics <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  ungroup() %>%
  mutate(x = n():1)

lda_a_post <- posterior(LDA_animation)
lda_a_json <- createJSON(
  phi = lda_a_post[['terms']],
  theta = lda_a_post[['topics']],
  vocab = colnames(lda_a_post[['terms']]),
  doc.length = slam::row_sums(LDA_animation@wordassignments, na.rm=TRUE),
  term.frequency = slam::col_sums(LDA_animation@wordassignments, na.rm=TRUE)
)

serVis(lda_a_json)
