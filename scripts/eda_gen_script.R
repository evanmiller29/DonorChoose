library(tidyverse)
library(ggplot2)
library(scales)

data_path <- "C:/Users/Evan/PycharmProjects/DonorChoose/data/"
graphics_path <- "C:/Users/Evan/PycharmProjects/DonorChoose/graphics/"

fig_width = 14
fig_height = 10

setwd(data_path)

# Resources ---------------------------------------------------------------

# Quantity ----------------------------------------------------------------

resources <- read_csv("resources.csv") %>% 
  mutate(ttl_cost = quantity * price) %>% 
  mutate(price_90_perc = quantile(price, prob=0.9), # Adding in percentiles to get a measure of how extreme the value is
         quant_90_perc = quantile(quantity, prob=0.9),
         ttl_cost_90_perc = quantile(price, prob=0.9)) %>% 
  mutate(abv_price_90_perc = price >= price_90_perc, # Creating greater than flags
         abv_quant_90_perc = quantity >= quant_90_perc,
         abv_ttl_cost_90_perc = ttl_cost >= ttl_cost_90_perc) 
  
glimpse(resources)
head(resources)

g1 <- resources %>% 
  filter(quantity < 100) %>% 
  ggplot(aes(x=quantity)) + geom_histogram(bins=100)

ggsave(paste0(graphics_path, "quant_less_hundred.png"), g1, height = fig_height, width = fig_width)

g1 <- resources %>% 
  filter(quantity >= 100) %>% 
  ggplot(aes(x=quantity)) + geom_histogram(bins=100)

print(max(resources$quantity))

ggsave(paste0(graphics_path, "quant_more_hundred.png"), g1, height = fig_height, width = fig_width)

# Price -------------------------------------------------------------------

g1 <- resources %>% 
  filter(price < 1000) %>% 
  ggplot(aes(x=price)) + geom_histogram(bins=100) +
  scale_y_continuous(labels=dollar)

ggsave(paste0(graphics_path, "price_less_thousand.png"), g1, height = fig_height, width = fig_width)

g1 <- resources %>% 
  filter(price >= 1000) %>% 
  ggplot(aes(x=price)) + geom_histogram(bins=100) +
  scale_x_continuous(breaks=seq(1000, 10000, by=1000), limits=c(0, 10000), labels=dollar)

ggsave(paste0(graphics_path, "price_more_thousand.png"), g1, height = fig_height, width = fig_width)

# Total cost --------------------------------------------------------------

g1 <- resources %>% 
  ggplot(aes(x=ttl_cost)) + geom_histogram(bins=100) +
  scale_y_continuous(breaks=seq(0, 1250000, by=250000), limits=c(0, 1250000), labels=comma)

ggsave(paste0(graphics_path, "ttl_cost.png"), g1, height = fig_height, width = fig_width)

g1 <- resources %>% 
  filter(ttl_cost > 1000) %>% 
  ggplot(aes(x=ttl_cost)) + geom_histogram(bins=100) + scale_y_continuous(labels=dollar)

ggsave(paste0(graphics_path, "ttl_cost_more_thou.png"), g1, height = fig_height, width = fig_width)
