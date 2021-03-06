#Using R in R Studio

Assignment#1
#Using EuStockMarkets data in R

#load the dataset in R
data()

#check the headers in dataset
head(EuStockMarkets)

mkt <- EuStockMarkets
head(mkt)

#linear fit the data
lin.fit0 <- lm(formula= FTSE ~ DAX, data=mkt)
summary(lin.fit0)
lin.fit1 <- lm(formula= FTSE ~ SMI, data=mkt)
summary(lin.fit1)
lin.fit2 <- lm(formula= FTSE ~ CAC, data=mkt)
summary(lin.fit2)
lin.fit3 <- lm(formula= FTSE ~ ., data=mkt)
summary(lin.fit3)

#BEST FIT is lin.fit1 based on the value of R-Squared
plot(lin.fit1)

#Polynomial Regression of EuStockMarkets 
#this is returning an error -$ operator is invalid for atomic vectors - will fix this once i know what to do
mktCol = as.data.frame(mkt)
mkt.polyfit <- lm(formula=mktCol$FTSE ~ poly(mktCol$SMI, 2, raw=TRUE ))
summary(mkt.polyfit)
mkt.polyfit1 <- lm(formula=mktCol$FTSE ~ poly(mktCol$SMI, 3, raw=TRUE ))
summary(mkt.polyfit1)


#Ridge Regression
rdg.ridge.fit <- lm.ridge(FTSE ~ .,data=mkt, lambda= seq(0,6000,1))
summary(rdg.ridge.fit)
plot(rdg.ridge.fit)

