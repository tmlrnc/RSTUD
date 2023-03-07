## ----echo=FALSE,eval=TRUE------------------------------------------------
options(continue="  ")

## ------------------------------------------------------------------------
options(digits=3)
options(width=72)  # narrows output to stay in the grey box
ds = read.csv("http://www.amherst.edu/~nhorton/r2/datasets/help.csv")

## ------------------------------------------------------------------------
with(ds, mean(cesd))

## ------------------------------------------------------------------------
with(ds, median(cesd))
with(ds, range(cesd))
with(ds, sd(cesd))
with(ds, var(cesd))
library(mosaic)
favstats(~ cesd, data=ds)

## ------------------------------------------------------------------------
library(moments)
with(ds, skewness(cesd))
with(ds, kurtosis(cesd))

## ------------------------------------------------------------------------
with(ds, quantile(cesd, seq(from=0, to=1, length=11)))

## ----bplotr,echo=TRUE,eval=TRUE------------------------------------------
with(ds, hist(cesd, main="", freq=FALSE))
with(ds, lines(density(cesd), main="CESD", lty=2, lwd=2))
xvals = with(ds, seq(from=min(cesd), to=max(cesd), length=100))
with(ds, lines(xvals, dnorm(xvals, mean(cesd), sd(cesd)), lwd=2))

## ----cormat1-------------------------------------------------------------
cormat = cor(with(ds, cbind(cesd, mcs, pcs)))
cormat

## ----cormat2-------------------------------------------------------------
cormat[c(2, 3), 1]

## ----rugplotr,echo=TRUE,eval=TRUE----------------------------------------
with(ds, plot(cesd[female==1], mcs[female==1], xlab="CESD", ylab="MCS",
   type="n", bty="n"))
with(ds, text(cesd[female==1&substance=="alcohol"],
   mcs[female==1&substance=="alcohol"],"A"))
with(ds, text(cesd[female==1&substance=="cocaine"],
   mcs[female==1&substance=="cocaine"],"C"))
with(ds, text(cesd[female==1&substance=="heroin"],
   mcs[female==1&substance=="heroin"],"H"))
with(ds, rug(jitter(mcs[female==1]), side=2))
with(ds, rug(jitter(cesd[female==1]), side=3))

## ----message=FALSE-------------------------------------------------------
require(gmodels)
with(ds, CrossTable(homeless, female, prop.chisq=FALSE, format="SPSS"))

## ------------------------------------------------------------------------
or = with(ds, (sum(homeless==0 & female==0)*
       sum(homeless==1 & female==1))/
      (sum(homeless==0 & female==1)*
       sum(homeless==1 & female==0)))
or

## ------------------------------------------------------------------------
library(epitools)
oddsobject = with(ds, oddsratio.wald(homeless, female))
oddsobject$measure
oddsobject$p.value

## ------------------------------------------------------------------------
chisqval = with(ds, chisq.test(homeless, female, correct=FALSE))
chisqval

## ------------------------------------------------------------------------
with(ds, fisher.test(homeless, female))

## ----gridtable,echo=TRUE, eval=TRUE,message=FALSE------------------------
library(gridExtra)
mytab = tally(~ racegrp + substance, data=ds)
plot.new()
grid.table(mytab)

## ------------------------------------------------------------------------
ttres = t.test(age ~ female, data=ds)
print(ttres)

## ----message=FALSE-------------------------------------------------------
library(coin)
oneway_test(age ~ as.factor(female),
   distribution=approximate(B=9999), data=ds)

## ------------------------------------------------------------------------
with(ds, wilcox.test(age ~ as.factor(female), correct=FALSE))

## ----kstest--------------------------------------------------------------
ksres = with(ds, ks.test(age[female==1], age[female==0]))
print(ksres)

## ------------------------------------------------------------------------
plotdens = function(x,y, mytitle, mylab) {
   densx = density(x)
   densy = density(y)
   plot(densx, main=mytitle, lwd=3, xlab=mylab, bty="l")
   lines(densy, lty=2, col=2, lwd=3)
   xvals = c(densx$x, rev(densy$x))
   yvals = c(densx$y, rev(densy$y))
   polygon(xvals, yvals, col="gray")
}

## ----polyplotr,echo=TRUE,eval=TRUE---------------------------------------
mytitle = paste("Test of ages: D=", round(ksres$statistic, 3),
   " p=", round(ksres$p.value, 2), sep="")
with(ds, plotdens(age[female==1], age[female==0], mytitle=mytitle, 
   mylab="age (in years)"))
legend(50, .05, legend=c("Women", "Men"), col=1:2, lty=1:2, lwd=2)

## ------------------------------------------------------------------------
library(survival)
survobj = survdiff(Surv(dayslink, linkstatus) ~ treat, 
   data=ds)
print(survobj)
names(survobj)



## ----echo=FALSE,eval=TRUE------------------------------------------------
options(continue="  ")

## ------------------------------------------------------------------------
options(digits=3)
ds = data.frame(x=rnorm(8), id=1:8)

## ------------------------------------------------------------------------
options(digits=3)
ds = ds[order(ds$x),]
ds

## ------------------------------------------------------------------------
diffx = diff(ds$x)
min(diffx)
with(ds, id[which.min(diffx)]) # first val
with(ds, id[which.min(diffx) + 1]) # second val

## ------------------------------------------------------------------------
p = .78 + (3 * 1:7)/100
allprobs = matrix(nrow=length(p), ncol=11)
for (i in 1:length(p)) {
   allprobs[i,] = round(dbinom(0:10, 10, p[i]),2)
}
table = cbind(p, allprobs)

## ------------------------------------------------------------------------
table

## ------------------------------------------------------------------------
runave = function(n, gendist, ...) {
   x = gendist(n, ...)
   avex = numeric(n)
   for (k in 1:n) {
      avex[k] = mean(x[1:k])
   }
   return(data.frame(x, avex))
}

## ------------------------------------------------------------------------
vals = 1000
set.seed(1984)
cauchy = runave(vals, rcauchy)
t4 = runave(vals, rt, 4)

## ----runaver,echo=TRUE, eval=TRUE----------------------------------------
plot(c(cauchy$avex, t4$avex), xlim=c(1, vals), type="n")
lines(1:vals, cauchy$avex, lty=1, lwd=2)
lines(1:vals, t4$avex, lty=2, lwd=2)
abline(0, 0)
legend(vals*.6, -1, legend=c("Cauchy", "t with 4 df"), 
   lwd=2, lty=c(1, 2))

## ------------------------------------------------------------------------
len = 10
fibvals = numeric(len)
fibvals[1] = 1
fibvals[2] = 1
for (i in 3:len) { 
   fibvals[i] = fibvals[i-1] + fibvals[i-2]
} 
fibvals

## ------------------------------------------------------------------------
# read in the data 
input = 
readLines("http://www.amherst.edu/~nhorton/r2/datasets/co25_d00.dat",
   n=-1)
# figure out how many counties, and how many entries
num = length(grep("END", input))
allvals = length(input)
numentries = allvals-num
# create vectors to store data
county = numeric(numentries); 
lat = numeric(numentries)
long = numeric(numentries)

## ------------------------------------------------------------------------
curval = 0   # number of counties seen so far
# loop through each line
for (i in 1:allvals) {
   if (input[i]=="END") {
      curval = curval + 1
   } else {
      # remove extraneous spaces
      nospace = gsub("[ ]+", " ", input[i])
      # remove space in first column
      nospace = gsub("^ ", "", nospace)
      splitstring = as.numeric(strsplit(nospace, " ")[[1]])
      len = length(splitstring)
      if (len==3) {  # new county
         curcounty = splitstring[1]; county[i-curval] = curcounty
         lat[i-curval] = splitstring[2]; long[i-curval] = splitstring[3]
      } else if (len==2) { # continue current county
         county[i-curval] = curcounty; lat[i-curval] = splitstring[1]
         long[i-curval] = splitstring[2]
      }
   }
}

## ------------------------------------------------------------------------
# read county names
countynames = 
read.table("http://www.amherst.edu/~nhorton/r2/datasets/co25_d00a.dat", 
   header=FALSE)
names(countynames) = c("county", "countyname")

## ------------------------------------------------------------------------
counties = unique(county)
xvals = c(min(lat), max(lat)); yvals = c(range(long))

## ----echo=TRUE, eval=TRUE------------------------------------------------
plot(xvals, yvals, pch=" ", xlab="", ylab="", xaxt="n", yaxt="n")
for (i in 1:length(counties)) {  # first element is an internal point
   polygon(lat[county==counties[i]][-1], long[county==counties[i]][-1])
   # plot name of county using internal point
   text(lat[county==counties[i]][1], long[county==counties[i]][1],
      countynames$countyname[i], cex=0.8)
}

## ----echo=FALSE----------------------------------------------------------
options(digits=4)

## ----message=FALSE-------------------------------------------------------
library(ggmap)
options(digits=4)
amherst = c(lon=-72.52, lat=42.36)
mymap = get_map(location=amherst, zoom=13, color="bw")

## ------------------------------------------------------------------------
myride =
   read.csv("http://www.amherst.edu/~nhorton/r2/datasets/cycle.csv")
head(myride, 2)

## ----bikeplotmap,echo=TRUE, eval=TRUE------------------------------------
ggmap(mymap) + geom_point(aes(x=Longitude, y=Latitude), data=myride)

## ----echo=FALSE----------------------------------------------------------
options(digits=3)

## ----choro0,echo=TRUE, eval=TRUE-----------------------------------------
library(ggmap); library(dplyr)
USArrests.st = mutate(USArrests, 
  region=tolower(rownames(USArrests)),
  murder = cut_number(Murder, 5))
us_state_map = map_data('state')
map_data = merge(USArrests.st, us_state_map, by="region")
map_data = arrange(map_data, order)
head(select(map_data, region, Murder, murder, long, lat, group, order))

## ----choro,echo=TRUE, eval=TRUE------------------------------------------
p0 = ggplot(map_data, aes(x=long, y=lat, group=group)) +
   geom_polygon(aes(fill = murder)) +
   geom_path(colour='black') +
   theme(legend.position = "bottom", 
      panel.background=element_rect(fill="transparent",
         color=NA)) +
   scale_fill_grey(start=1, end =.1) + coord_map();
plot(p0)

## ----eval=FALSE, echo=TRUE-----------------------------------------------
## # grab contents of web page
## urlcontents = readLines("http://tinyurl.com/cartoonguide")
## 
## # find line with sales rank
## linenum = suppressWarnings(grep("See Top 100 in Books", urlcontents))
## 
## # split line into multiple elements
## linevals = strsplit(urlcontents[linenum], ' ')[[1]]
## 
## # find element with sales rank number
## entry = grep("#", linevals)
## charrank = linevals[entry] # snag that entry
## charrank = substr(charrank, 2, nchar(charrank)) # kill '#' at start
## charrank = gsub(',' ,'', charrank) # remove commas
## salesrank = as.numeric(charrank) # make it numeric
## cat("salesrank=", salesrank, "\n")

## ----message=FALSE-------------------------------------------------------
library(RCurl)
myurl = 
   getURL("https://www3.amherst.edu/~nhorton/r2/datasets/cartoon.txt",
             ssl.verifypeer=FALSE)
file = readLines(textConnection(myurl))
n = length(file)/2
rank = numeric(n)
timeval = as.POSIXlt(rank, origin="1960-01-01")
for (i in 1:n) {
   timeval[i] = as.POSIXlt(gsub('EST', '', 
      gsub('EDT', '', file[(i-1)*2+1])), 
      tz="EST5EDT", format="%a %b %d %H:%M:%S %Y")
   rank[i] = as.numeric(gsub('NA', '', 
      gsub('salesrank= ','', file[i*2])))
}
timerank = data.frame(timeval, rank)

## ------------------------------------------------------------------------
head(timerank, 4)

## ----message=FALSE-------------------------------------------------------
library(lubridate)
timeofday = hour(timeval)
night = rep(0,length(timeofday))  # vector of zeroes
night[timeofday < 8 | timeofday > 18] = 1

## ----cartoonplot,eval=TRUE, echo=TRUE------------------------------------
plot(timeval, rank, type="l", xlab="", ylab="Amazon Sales Rank")
points(timeval[night==1], rank[night==1], pch=20, col="black")
points(timeval[night==0], rank[night==0], pch=20, col="red")
legend(as.POSIXlt("2013-10-03 00:00:00 EDT"), 6000, 
   legend=c("day","night"), col=c("red","black"), pch=c(20,20))
abline(v=as.numeric(as.POSIXlt("2013-10-01 00:00:00 EST")), lty=2)

## ----message=FALSE-------------------------------------------------------
require(XML)
require(mosaic)
wikipedia = "http://en.wikipedia.org/wiki"
liverpool = "List_of_films_and_television_shows_set_or_shot_in_Liverpool"
result = readHTMLTable(paste(wikipedia, liverpool, sep="/"),
  stringsAsFactors=FALSE)
table1 = result[[2]]
names(table1)

## ------------------------------------------------------------------------
require(dplyr)
finaltable = table1 %>% 
  mutate(year = as.numeric(Year)) %>%
  select(year, Title) 
head(finaltable, 8)
favstats(~ year, data=finaltable)

## ------------------------------------------------------------------------
with(finaltable, stem(year, scale=2))

## ------------------------------------------------------------------------
truerand = function(numrand) {
   read.table(as.character(paste("http://www.random.org/integers/?num=", 
   numrand, "&min=0&max=1000000000&col=1&base=10&format=plain&rnd=new", 
   sep="")))/1000000000
}

quotacheck = function() {
   line = as.numeric(readLines(
      "http://www.random.org/quota/?format=plain"))
   return(line)
}

## ------------------------------------------------------------------------
truerand(7)
quotacheck()

## ----echo=FALSE----------------------------------------------------------
library(httr)
load("questions.Rda")   # silently load

## ----eval=FALSE----------------------------------------------------------
## library(httr)
## # Find the most recent R questions on stackoverflow
## getresult = GET("http://api.stackexchange.com",
##   path="questions",
##   query=list(site="stackoverflow.com", tagged="dplyr"))
## stop_for_status(getresult) # Ensure returned without error
## questions = content(getresult)  # Grab content

## ------------------------------------------------------------------------
names(questions$items[[1]])    # What does the returned data look like?
substr(questions$items[[1]]$title, 1, 68)
substr(questions$items[[2]]$title, 1, 68)
substr(questions$items[[3]]$title, 1, 68)

## ------------------------------------------------------------------------
library(aRxiv)
library(lubridate)
library(stringr)
library(dplyr)
efron = arxiv_search(query='au:"Efron" AND cat:stat*', limit=50)
names(efron)
dim(efron)
efron = mutate(efron, submityear =
  year(sapply(str_split(submitted, " "), "[[", 1)))
with(efron, table(submityear))

## ----message=FALSE-------------------------------------------------------
library(tm)
mycorpus = VCorpus(DataframeSource(data.frame(efron$abstract)))
head(strwrap(mycorpus[[1]]))

## ------------------------------------------------------------------------
mycorpus = tm_map(mycorpus, stripWhitespace)
mycorpus = tm_map(mycorpus, removeNumbers)
mycorpus = tm_map(mycorpus, removePunctuation)
mycorpus = tm_map(mycorpus, content_transformer(tolower))
mycorpus = tm_map(mycorpus, removeWords, stopwords("english"))
head(strwrap(mycorpus[[1]]))

## ------------------------------------------------------------------------
dtm = DocumentTermMatrix(mycorpus)
findFreqTerms(dtm, 7)

## ----eval=FALSE,prompt=FALSE---------------------------------------------
## library(ggvis)
## > ds = read.csv("http://www.amherst.edu/~nhorton/r2/datasets/help.csv")
## > ds %>%
##   ggvis(x = ~ mcs, y = ~ cesd,
##     size := input_slider(min=10, max=100, label="size"),
##     opacity := input_slider(min=0, max=1, label="opacity"),
##     fill := input_select(choices=c("red", "green", "blue", "grey"),
##       selected="red", label="fill color"),
##     stroke := "black") %>%
##   layer_points()

## ----eval=FALSE, prompt=FALSE--------------------------------------------
## ---
## title: "Sample Shiny in Markdown"
## output: html_document
## runtime: shiny
## ---
## 
## Shiny inputs and outputs can be embedded in a Markdown document.  Outputs
## are automatically updated whenever inputs change.  This demonstrates
## how a standard R plot can be made interactive by wrapping it in the
## Shiny `renderPlot` function. The `selectInput` function creates the
## input widgets used to control the plot display.
## 

## Note changes from book (due to revised interface in choroplethr package)

## ui.R
## ------------------
## library(shiny)
## shinyUI(bootstrapPage(
##  selectInput("n_breaks", label = "Number of breaks:",
##    choices = c(1, 2, 3, 4, 5, 9), selected = 5),
##  selectInput("labels", label = "Display labels?",
##    choices = c("TRUE", "FALSE"), selected = "TRUE"),
##  plotOutput(outputId = "main_plot", height = "300px", width="500px")
## ))


## session.R
## ------------------
## shinyServer(function(input, output) {
##   output$main_plot <- renderPlot({
##     library(choroplethr); library(choroplethrMaps); library(dplyr)
##     USArrests.st = mutate(USArrests,
##       region=tolower(rownames(USArrests)),
##       value = Murder) 
##     USArrests.st = select(USArrests.st, region, value)
##     choro = StateChoropleth$new(USArrests.st)
##     choro$show_labels = input$labels
##     choro$title="Murder Rates by State"
##     choro$set_buckets(as.numeric(input$n_breaks))
##     choro$render()
##   })
## })

## > shinyApp(ui=ui, server=server)

## ----eval=FALSE----------------------------------------------------------
## library(shiny)
## runApp("~/ShinyApps/choropleth")

## ----eval=FALSE----------------------------------------------------------
## library(RSQLite)
## con = dbConnect("SQLite", dbname = "/Home/Airlines/ontime.sqlite3")
## ds = dbGetQuery(con, "SELECT DayofMonth, Month, Year, Origin,
##    sum(1) as numFlights FROM ontime WHERE Origin='BDL'
##    GROUP BY DayofMonth,Month,Year")
## # returns a dataframe with 7,763 rows and 5 columns

## ----eval=FALSE----------------------------------------------------------
## library(dplyr)
## ds = mutate(ds, date =
##    as.Date(paste(Year, "-", Month, "-", DayofMonth, sep="")))
## ds = mutate(ds, weekday = weekdays(date))
## ds = arrange(ds, date)
## mondays = filter(ds, weekday=="Monday")

## ----eval=FALSE----------------------------------------------------------
## library(lattice)
## xyplot(numFlights ~ date, xlab="", ylab="number of flights on Monday",
##    type="l", col="black", lwd=2, data=mondays)

## ----eval=FALSE, prompt=FALSE--------------------------------------------
## > library(dplyr)
## > my_db = src_sqlite("/Home/Airlines/ontime.sqlite3")
## > my_tbl = group_by(tbl(my_db, "ontime"), DayofMonth, Month, Year, Origin)
## > ds = my_tbl %>%
##   filter(Origin=="BDL") %>%
##   select(DayofMonth, Month, Year, Origin) %>%
##   summarise(numFlights=n())

## ------------------------------------------------------------------------
# Define constants and useful functions
weight = c(0.3, 0.2, 2.0)
volume = c(2.5, 1.5, 0.2)
value = c(3000, 1800, 2500)
maxwt = 25
maxvol = 25

## ------------------------------------------------------------------------
# minimize the grid points we need to calculate
max.items = floor(pmin(maxwt/weight, maxvol/volume))

# useful functions
getvalue = function(n) sum(n*value)
getweight = function(n) sum(n*weight)
getvolume = function(n) sum(n*volume)

# main function: return 0 if constraints not met,
# otherwise return the value of the contents, and their weight
findvalue = function(x) {
   thisweight = apply(x, 1, getweight)
   thisvolume = apply(x, 1, getvolume)
   fits = (thisweight <= maxwt) &
          (thisvolume <= maxvol)
   vals = apply(x, 1, getvalue)
   return(data.frame(panacea=x[,1], ichor=x[,2], gold=x[,3],
       value=fits*vals, weight=thisweight, 
       volume=thisvolume))
}
  
# Find and evaluate all possible combinations
combs = expand.grid(lapply(max.items, function(n) seq.int(0, n)))
values = findvalue(combs)

## ------------------------------------------------------------------------
max(values$value)
values[values$value==max(values$value),]

