### Percent of votes (total votes) from registered electors

```{r percent_of_total_voters}
pal <- colorBin("Oranges", presence_n$Total.Percent, bins = 8, na.color = "Red")
#initialize popup
countyPopup <- paste0("<h3>Judet (County):&nbsp<font color=\"blue\">",presence_n$Name,"</font></h3>",
                      "Total electors in the county:&nbsp<b><font color=\"blue\">",presence_n$Total.Electors,"</font></b>",
                      "<br><br></strong>Percent of voters:&nbsp<b><font color=\"red\">",round(presence_n$Total.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of registered voters:&nbsp<b><font color=\"blue\">",round(presence_n$Registered.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of non-registered voters:&nbsp<b><font color=\"blue\">",round(presence_n$Not.Registered.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of remote voters:&nbsp<b><font color=\"blue\">",round(presence_n$Mobile.Percent,2),"%</font></b>")

#prepare leaflet
leaflet(data = rgeojson) %>% 
  addTiles() %>%
  setView(25.6, 45.6, 7) %>% 
  addPolygons(fillColor = ~pal(presence_n$Total.Percent), 
              fillOpacity = 0.9, 
              color = "#7700BB", 
              weight = 1, 
              popup = countyPopup) %>%
  addLegend("topright", pal = pal, 
            values = c(min(presence_n$presence_n),
                       max(presence_n$Total.Percent)),
            title = "Percent voters<br> from registered total electors",
            labFormat = labelFormat(suffix = "%"),
            opacity = 1
  )
```

### Percent of votes cast from non-registered electors


```{r percent_of_non_registered_voters}
pal <- colorBin("Reds", presence_n$Not.Registered.Percent, bins = 8, na.color = "Red")
#initialize popup
countyPopup <- paste0("<h3>Judet (County):&nbsp<font color=\"blue\">",presence_n$Name,"</font></h3>",
                      "Total electors in the county:&nbsp<b><font color=\"blue\">",presence_n$Total.Electors,"</font></b>",
                      "<br><br></strong>Percent of voters:&nbsp<b><font color=\"blue\">",round(presence_n$Total.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of registered voters:&nbsp<b><font color=\"blue\">",round(presence_n$Registered.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of non-registered voters:&nbsp<b><font color=\"red\">",round(presence_n$Not.Registered.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of remote voters:&nbsp<b><font color=\"blue\">",round(presence_n$Mobile.Percent,2),"%</font></b>")

#prepare leaflet
leaflet(data = rgeojson) %>% 
  addTiles() %>%
  setView(25.6, 45.6, 7) %>% 
  addPolygons(fillColor = ~pal(presence_n$Not.Registered.Percent), 
              fillOpacity = 0.9, 
              color = "#7700BB", 
              weight = 1, 
              popup = countyPopup) %>%
  addLegend("topright", pal = pal, 
            values = c(min(presence_n$Not.Registered.Percent),
                       max(presence_n$Not.Registered.Percent)),
            title = "Percent non-registered voters<br> from registered total electors",
            labFormat = labelFormat(suffix = "%"),
            opacity = 1
  )
```


### Percent of votes cast remotely (Rom: "urna mobila")

```{r percent_of_votes_cast_remotely}
pal <- colorBin("Reds", presence_n$Mobile.Percent, bins = 8, na.color = "Red")
#initialize popup
countyPopup <- paste0("<h3>Judet (County):&nbsp<font color=\"blue\">",presence_n$Name,"</font></h3>",
                      "Total electors in the county:&nbsp<b><font color=\"blue\">",presence_n$Total.Electors,"</font></b>",
                      "<br><br></strong>Percent of voters:&nbsp<b><font color=\"blue\">",round(presence_n$Total.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of registered voters:&nbsp<b><font color=\"blue\">",round(presence_n$Registered.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of non-registered voters:&nbsp<b><font color=\"blue\">",round(presence_n$Not.Registered.Percent,2),"%</font></b>",
                      "<br><br></strong>Percent of remote voters:&nbsp<b><font color=\"red\">",round(presence_n$Mobile.Percent,2),"%</font></b>")

#prepare leaflet
leaflet(data = rgeojson) %>% 
  addTiles() %>%
  setView(25.6, 45.6, 7) %>% 
  addPolygons(fillColor = ~pal(presence_n$Mobile.Percent), 
              fillOpacity = 0.9, 
              color = "#7700BB", 
              weight = 1, 
              popup = countyPopup) %>%
  addLegend("topright", pal = pal, 
            values = c(min(presence_n$Mobile.Percent),
                       max(presence_n$Mobile.Percent)),
            title = "Percent remote voters<br> from registered total electors",
            labFormat = labelFormat(suffix = "%"),
            opacity = 1
  )
```