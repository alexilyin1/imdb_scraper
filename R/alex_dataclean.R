library(tidyverse)
library(readr)
library(plyr)

folders <- c(list.files('./data/images', full.names = TRUE))
images <- lapply(folders, function(x) {
  list.files(x)
})
images <- as.data.frame(matrix(unlist(images)), nrow=length(unlist(images[1]))) %>%
  mutate(V1 = as.character(V1))
images <- str_remove(images$V1, '.png')
images <- str_remove(images, 'tt')

old_reviews <- readRDS('~/git/hamsum-hasen/all_reviews.rds') 
reviews_full <- old_reviews %>%
  filter(V1 %in% images)

# setwd('../')
# saveRDS(reviews_full, 'full_reviews.rds')

reviews_full$genre <- ifelse(str_c('tt', reviews_full$V1, '.png') %in% list.files('data/images/Animation/'),
                             'Animation', 
                                ifelse(str_c('tt', reviews_full$V1, '.png') %in% list.files('data/images/Sci/'),
                                'Sci-Fi', 
                                    ifelse(str_c('tt', reviews_full$V1, '.png') %in% list.files('data/images/War'),
                                    'War',
                                        ifelse(str_c('tt', reviews_full$V1, '.png') %in% list.files('data/images/Western/'),
                                        'Western', ''))))


a.reviews <- filter(reviews_full, genre=='Animation')
s.reviews <- filter(reviews_full, genre=='Sci-Fi')
wa.reviews <- filter(reviews_full, genre=='War')
we.reviews <- filter(reviews_full, genre=='Western')

write.csv(a.reviews, 'a_reviews.csv')
write.csv(s.reviews, 's_reviews.csv')
write.csv(wa.reviews, 'wa_reviews.csv')
write.csv(we.reviews, 'we_reviews.csv')
