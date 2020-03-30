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

s <- read_csv('./s_reviews.csv') %>%
  select(-c(X1))

wa <- read_csv('./wa_reviews.csv') %>%
  select(-c(X1))

we <- read_csv('./we_reviews.csv') %>%
  select(-c(X1))


#*********************************************************************************************
## Topic models for 'animation' genre

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

  
topics_a <- tidy(LDA_animation, matrix = 'beta')

# Topic Visualizations
top_topics_a <- topics_a %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  ungroup() %>%
  mutate(x = n():1)

top_topics_a %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill=factor(topic))) +
  geom_col(show.legend=FALSE) + facet_wrap(~topic, scales='free') +
  coord_flip() + scale_x_reordered()

lda_a_post <- posterior(LDA_animation)
lda_a_json <- createJSON(
  phi = lda_a_post[['terms']],
  theta = lda_a_post[['topics']],
  vocab = colnames(lda_a_post[['terms']]),
  doc.length = slam::row_sums(LDA_animation@wordassignments, na.rm=TRUE),
  term.frequency = slam::col_sums(LDA_animation@wordassignments, na.rm=TRUE)
)

serVis(lda_a_json)


#*********************************************************************************************
## Topic models for Western movies

western_tidy <- we %>%
  select(V1, V2) %>%
  unnest_tokens(word, V2) %>%
  anti_join(stop_words)

western_tidy <- western_tidy %>%
  mutate(lemma = lemmatize_words(word)) 

wordCount <- western_tidy %>%
  group_by(word) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

wordCount <- western_tidy %>%
  group_by(lemma) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  filter(count > 20)

final_words <- western_tidy %>%
  filter(lemma %in% wordCount$lemma) %>%
  select(V1, lemma)

dtm_w <- western_tidy %>%
  count(V1, lemma) %>%
  cast_dtm(V1, lemma, n)

## LDA model
LDA_western <- LDA(dtm_w, k = 15, method = 'Gibbs',
                   control = list(alpha = 1/10, iter=50, 
                                  burnin=10000, seed = 1234))


topics_w <- tidy(LDA_western, matrix = 'beta')

# Topic visualizations
top_topics_w <- topics_w %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  ungroup() %>%
  mutate(x = n():1)

top_topics_w %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill=factor(topic))) +
  geom_col(show.legend=FALSE) + facet_wrap(~topic, scales='free') +
  coord_flip() + scale_x_reordered()

lda_w_post <- posterior(LDA_western)
lda_w_json <- createJSON(
  phi = lda_w_post[['terms']],
  theta = lda_w_post[['topics']],
  vocab = colnames(lda_w_post[['terms']]),
  doc.length = slam::row_sums(LDA_western@wordassignments, na.rm=TRUE),
  term.frequency = slam::col_sums(LDA_western@wordassignments, na.rm=TRUE)
)

serVis(lda_w_json)
  

#*********************************************************************************************
## Topic models for War movies

war_tidy <- wa %>%
  select(V1, V2) %>%
  unnest_tokens(word, V2) %>%
  anti_join(stop_words)

war_tidy <- war_tidy %>%
  mutate(lemma = lemmatize_words(word)) 

wordCount <- war_tidy %>%
  group_by(word) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

wordCount <- war_tidy %>%
  group_by(lemma) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  filter(count > 20)

final_words <- war_tidy %>%
  filter(lemma %in% wordCount$lemma) %>%
  select(V1, lemma)

dtm_wa <- war_tidy %>%
  count(V1, lemma) %>%
  cast_dtm(V1, lemma, n)

## LDA model
LDA_war <- LDA(dtm_wa, k = 15, method = 'Gibbs',
                   control = list(alpha = 1/10, iter=50, 
                                  burnin=10000, seed = 1234))


topics_wa <- tidy(LDA_war, matrix = 'beta')

# Topic Visualizations
top_topics_wa <- topics_wa %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  ungroup() %>%
  mutate(x = n():1)

top_topics_wa %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill=factor(topic))) +
  geom_col(show.legend=FALSE) + facet_wrap(~topic, scales='free') +
  coord_flip() + scale_x_reordered()

lda_wa_post <- posterior(LDA_war)
lda_wa_json <- createJSON(
  phi = lda_wa_post[['terms']],
  theta = lda_wa_post[['topics']],
  vocab = colnames(lda_wa_post[['terms']]),
  doc.length = slam::row_sums(LDA_war@wordassignments, na.rm=TRUE),
  term.frequency = slam::col_sums(LDA_war@wordassignments, na.rm=TRUE)
)

serVis(lda_wa_json)


#*********************************************************************************************
## Topic models for Sci-Fi movies

sci_tidy <- s %>%
  select(V1, V2) %>%
  unnest_tokens(word, V2) %>%
  anti_join(stop_words)

sci_tidy <- sci_tidy %>%
  mutate(lemma = lemmatize_words(word)) 

wordCount <- sci_tidy %>%
  group_by(word) %>%
  summarise(count = n()) %>%
  arrange(desc(count))

wordCount <- sci_tidy %>%
  group_by(lemma) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  filter(count > 20)

final_words <- sci_tidy %>%
  filter(lemma %in% wordCount$lemma) %>%
  select(V1, lemma)

dtm_sci <- sci_tidy %>%
  count(V1, lemma) %>%
  cast_dtm(V1, lemma, n)

## LDA model
LDA_sci <- LDA(dtm_sci, k = 15, method = 'Gibbs',
               control = list(alpha = 1/10, iter=50, 
                              burnin=10000, seed = 1234))


topics_sci <- tidy(LDA_sci, matrix = 'beta')

# Topic visualizations 
top_topics_sci <- topics_sci %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  ungroup() %>%
  mutate(x = n():1)

top_topics_sci %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill=factor(topic))) +
  geom_col(show.legend=FALSE) + facet_wrap(~topic, scales='free') +
  coord_flip() + scale_x_reordered()

lda_s_post <- posterior(LDA_sci)
lda_s_json <- createJSON(
  phi = lda_s_post[['terms']],
  theta = lda_s_post[['topics']],
  vocab = colnames(lda_s_post[['terms']]),
  doc.length = slam::row_sums(LDA_sci@wordassignments, na.rm=TRUE),
  term.frequency = slam::col_sums(LDA_sci@wordassignments, na.rm=TRUE)
)

serVis(lda_s_json)


## Gamma probability 
gamma_a <- tidy(LDA_animation, matrix='gamma')
gamma_a

a %>%
  filter(V1==1049413)


## Sentiment analysis
install.packages('sentimentr')
library(sentimentr)

sentences <- get_sentences(a$V2)
a.sentiment <- sentiment_by(sentences)

a$element_id <- seq(1, nrow(a))

animation_sentiments <- left_join(a, a.sentiment, by=c('element_id'))

ggplot(animation_sentiments, aes(ave_sentiment)) + geom_histogram()

animation_sentiments$V1 <- as.character(animation_sentiments$V1)
animation_sentiments <- left_join(animation_sentiments, gamma_a, by=c('V1'='document'))

genres_sentiment <- animation_sentiments %>%
  group_by(genre) %>%
  summarise(avg_sentiment = mean(ave_sentiment))

ggplot(top_topics, aes(topic)) + geom_histogram()
