

```{r presence_vote_final, fig.width=9}
presence_n %>%
  ggplot(aes(x=Registered.Percent, y=Total.Percent)) + guides(fill=FALSE) +
  geom_point() + geom_smooth(method=lm) + theme_bw() + 
  scale_x_continuous(limits = c(0, 35)) + scale_y_continuous(limits = c(0,35)) + 
  geom_label_repel(aes(label = Name),box.padding=0.15, point.padding=0.25,segment.color = 'blue') +
  labs(x="Registered votes from registered electors [%]", y="Total votes from registered electors [%]", 
       title="Percent of votes from registered electors", subtitle="Romanian Referendum 2018 - Rural and Urban areas")
```


```{r county_map_prepare,warnings=FALSE}
presence %>% filter(Date.Hour == max(Date.Hour)) %>% filter(Medium == 'R') %>% 
  group_by(County) %>%
  summarise(Registered.Percent = 100* sum(Votes.Registered)/sum(Electors),
            Not.Registered.Percent = 100 * sum(Votes.not.Registered)/sum(Electors),
            Mobile.Percent = 100 * sum(Votes.Mobile) / sum(Electors),
            Total.Percent = 100 * sum(Votes.Total)/sum(Electors),
            Total.Votes = sum(Votes.Total)) %>% ungroup() -> presence_county_rural

```

```{r county_map_prepare2,warnings=FALSE}
presence_r = merge(presence_county_rural,county_names_codes, by.x="County", by.y="Code")
presence_r = presence_r[matchingNames(rgeojson$name,as.character(presence_r$Name)),]
```

```{r presence_vote_final, fig.width=9}
presence_r %>%
  ggplot(aes(x=Registered.Percent, y=Total.Percent)) + guides(fill=FALSE) +
  geom_point() + geom_smooth(method=lm) + theme_bw() + 
  scale_x_continuous(limits = c(0, 35)) + scale_y_continuous(limits = c(0,35)) + 
  geom_label_repel(aes(label = Name),box.padding=0.15, point.padding=0.25,segment.color = 'blue') +
  labs(x="Registered votes from registered electors [%]", y="Total votes from registered electors [%]", 
       title="Percent of votes from registered electors", subtitle="Romanian Referendum 2018 - Rural areas")
```