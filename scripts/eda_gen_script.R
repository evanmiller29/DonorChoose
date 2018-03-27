library(tidyverse)
library(ggplot2)
library(scales)

data_path <- "C:/Users/Evan/PycharmProjects/DonorChoose/data/"
graphics_path <- "C:/Users/Evan/PycharmProjects/DonorChoose/graphics/"

fig_width = 14
fig_height = 10

setwd(data_path)

# Resources ---------------------------------------------------------------

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

# Quantity ----------------------------------------------------------------

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
  scale_y_continuous(labels=comma) + scale_x_continuous(labels=dollar)

ggsave(paste0(graphics_path, "price_less_thousand.png"), g1, height = fig_height, width = fig_width)

g1 <- resources %>% 
  filter(price >= 1000) %>% 
  ggplot(aes(x=price)) + geom_histogram(bins=100) +
  scale_x_continuous(breaks=seq(1000, 10000, by=1000), limits=c(0, 10000), labels=comma) +
  scale_x_continuous(labels=dollar)

ggsave(paste0(graphics_path, "price_more_thousand.png"), g1, height = fig_height, width = fig_width)

# Total cost --------------------------------------------------------------

g1 <- resources %>% 
  ggplot(aes(x=ttl_cost)) + geom_histogram(bins=100) +
  scale_y_continuous(breaks=seq(0, 1250000, by=250000), limits=c(0, 1250000), labels=comma) +
  scale_x_continuous(labels=dollar)

ggsave(paste0(graphics_path, "ttl_cost.png"), g1, height = fig_height, width = fig_width)

g1 <- resources %>% 
  filter(ttl_cost > 1000) %>% 
  ggplot(aes(x=ttl_cost)) + geom_histogram(bins=100) + scale_y_continuous(labels=comma) +
  scale_x_continuous(labels=dollar)

ggsave(paste0(graphics_path, "ttl_cost_more_thou.png"), g1, height = fig_height, width = fig_width)


# Train -------------------------------------------------------------------

train <- read_csv("train.csv")

glimpse(train)

# Number of unique teachers

train$teacher_id %>% unique() %>% length()

# Who submitted the most applications?

graph_df <- train %>% 
  group_by(teacher_id) %>% 
  summarise(count=n()) %>% 
  top_n(n = 10, count)
  
g1 <- ggplot(graph_df, aes(x=reorder(teacher_id, count), y=count)) + 
  geom_col() + coord_flip()

ggsave(paste0(graphics_path, "top_ten_teachers_by_apps.png"), g1, height = fig_height, width = fig_width)

# Approval rates by teacher - lots of people just apply once and get approved -------------------------

graph_df <- train %>% 
  group_by(teacher_id) %>% 
  summarise(ttl_approved= sum(project_is_approved),
            num = n()) %>% 
  mutate(prop_approved = ttl_approved / num) %>% 
  ungroup() %>% 
  group_by(prop_approved) %>% 
  summarise(count=n())

g1 <- ggplot(graph_df, aes(x=prop_approved, y=count)) + geom_line() +
  scale_x_continuous(labels=percent)

ggsave(paste0(graphics_path, "teacher_approval_rates.png"), g1, height = fig_height, width = fig_width)

# Approval rates by state -------------------------------------------------

graph_df <- train %>% 
  group_by(school_state) %>% 
  summarise(ttl_approved= sum(project_is_approved),
            num = n()) %>% 
  mutate(prop_approved = ttl_approved / num) %>% 
  mutate(avg_approved = mean(prop_approved)) %>% 
  mutate(above = prop_approved > avg_approved)

g1 <- ggplot(graph_df, aes(x=prop_approved, y=reorder(school_state, prop_approved), col=above)) +
  geom_segment(aes(x = 0, 
                   y = reorder(school_state, prop_approved), 
                   xend = prop_approved, 
                   yend = reorder(school_state, prop_approved)),
               color = "grey50") +
  geom_point()

ggsave(paste0(graphics_path, "state_approval_lollipop_plot.png"), g1, height = fig_height, width = fig_width)

# Approval rates by grade  --------------------------------------------------

graph_df <- train %>% 
  group_by(project_grade_category) %>% 
  summarise(ttl_approved= sum(project_is_approved),
            num = n()) %>% 
  mutate(prop_approved = ttl_approved / num) %>% 
  mutate(avg_approved = mean(prop_approved)) %>% 
  mutate(above = prop_approved > avg_approved)

g1 <- ggplot(graph_df, aes(x=prop_approved, y=reorder(project_grade_category, prop_approved), col=above)) +
  geom_segment(aes(x = 0, 
                   y = reorder(project_grade_category, prop_approved), 
                   xend = prop_approved, 
                   yend = reorder(project_grade_category, prop_approved)),
               color = "grey50") +
  geom_point()

ggsave(paste0(graphics_path, "grade_approval_lollipop_plot.png"), g1, height = fig_height, width = fig_width)


# Approval rates by subject category --------------------------------------

graph_df <- train %>% 
  group_by(project_subject_categories) %>% 
  summarise(ttl_approved= sum(project_is_approved),
            num = n()) %>% 
  mutate(prop_approved = ttl_approved / num) %>% 
  mutate(avg_approved = mean(prop_approved)) %>% 
  mutate(above = prop_approved > avg_approved)

g1 <- ggplot(graph_df, aes(x=prop_approved, y=reorder(project_subject_categories, prop_approved), col=above)) +
  geom_segment(aes(x = 0, 
                   y = reorder(project_subject_categories, prop_approved), 
                   xend = prop_approved, 
                   yend = reorder(project_subject_categories, prop_approved)),
               color = "grey50") +
  geom_point()

ggsave(paste0(graphics_path, "subject_approval_lollipop_plot.png"), g1, height = fig_height, width = fig_width)

# Approval rates by subject sub-category top/bottom 20 with more than 100 apps ----------------------------------

graph_df <- train %>% 
  group_by(project_subject_subcategories) %>% 
  summarise(ttl_approved= sum(project_is_approved),
            num = n()) %>% 
  mutate(prop_approved = ttl_approved / num) %>% 
  mutate(avg_approved = mean(prop_approved)) %>% 
  mutate(above = prop_approved > avg_approved)

graph_top_subcats <- graph_df %>% 
  filter(num > 100) %>% 
  arrange(desc(prop_approved)) %>% 
  slice(1:20)

graph_bottom_subcats <- graph_df %>% 
  filter(num > 100) %>% 
  arrange(prop_approved) %>% 
  slice(1:20)

g1 <- ggplot(graph_top_subcats, aes(x=prop_approved, y=reorder(project_subject_subcategories, prop_approved), col=above)) +
  geom_segment(aes(x = 0, 
                   y = reorder(project_subject_subcategories, prop_approved), 
                   xend = prop_approved, 
                   yend = reorder(project_subject_subcategories, prop_approved)),
               color = "grey50") +
  geom_point()

ggsave(paste0(graphics_path, "subcat_top_20_100_apps_approval_.png"), g1, height = fig_height, width = fig_width)

g1 <- ggplot(graph_bottom_subcats, aes(x=prop_approved, y=reorder(project_subject_subcategories, prop_approved), col=above)) +
  geom_segment(aes(x = 0, 
                   y = reorder(project_subject_subcategories, prop_approved), 
                   xend = prop_approved, 
                   yend = reorder(project_subject_subcategories, prop_approved)),
               color = "grey50") +
  geom_point()

ggsave(paste0(graphics_path, "subcat_bot_20_100_apps_approval_.png"), g1, height = fig_height, width = fig_width)
