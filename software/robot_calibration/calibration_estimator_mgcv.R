library(mgcv)
library(scam)
library(qgam)

#load data
xPushDF <- read.csv(file="/home/cns/Desktop/Trodes/linux/xPushDF.csv", header=TRUE, sep=",")
xPullDF <- read.csv(file="/home/cns/Desktop/Trodes/linux/xPullDF.csv", header=TRUE, sep=",")
yPushDF <- read.csv(file="/home/cns/Desktop/Trodes/linux/yPushDF.csv", header=TRUE, sep=",")
yPullDF <- read.csv(file="/home/cns/Desktop/Trodes/linux/yPullDF.csv", header=TRUE, sep=",")

#preprocess data (convert to bits, microseconds, etc.)
xPushDF$displacement <- xPushDF$displacement/5*1023
xPullDF$displacement <- -1*xPullDF$displacement/5*1023
yPushDF$displacement <- yPushDF$displacement/5*1023
yPullDF$displacement <- -1*yPullDF$displacement/5*1023
xPushDF$displacement[xPushDF$displacement<0] <- 0
xPullDF$displacement[xPullDF$displacement<0] <- 0
yPushDF$displacement[yPushDF$displacement<0] <- 0
yPullDF$displacement[yPullDF$displacement<0] <- 0
xPushDF$xposition <- xPushDF$xposition/5*1023
xPullDF$xposition <- xPullDF$xposition/5*1023
yPushDF$yposition <- yPushDF$yposition/5*1023
yPullDF$yposition <- yPullDF$yposition/5*1023
xPushDF$duration <- xPushDF$duration*1000000
xPullDF$duration <- xPullDF$duration*1000000
yPushDF$duration <- yPushDF$duration*1000000
yPullDF$duration <- yPullDF$duration*1000000

#fit gams
xPushGam <- gam(displacement ~ te(xposition,duration,k=c(10,10),bs=c("cr","cr")),data=xPushDF)
xPullGam <- gam(displacement ~ te(xposition,duration,k=c(10,10),bs=c("cr","cr")),data=xPullDF)
yPushGam <- gam(displacement ~ te(yposition,duration,k=c(10,10),bs=c("cr","cr")),data=yPushDF)
yPullGam <- gam(displacement ~ te(yposition,duration,k=c(10,10),bs=c("cr","cr")),data=yPullDF)

#plot gams
plot(xPushGam,residuals = FALSE,scheme=2)
plot(xPullGam,residuals = FALSE,scheme=2)
plot(yPushGam,residuals = FALSE,scheme=2)
plot(yPullGam,residuals = FALSE,scheme=2)

#get gam predictions
pdata <- data.frame("xposition" = rep(seq(min(xPushDF$xposition),max(xPushDF$xposition),length.out=50),times=50),
                    "duration" = rep(seq(1000,max(xPushDF$duration),length.out=50),each=50))
xPushGamPred <- predict(xPushGam,pdata,type="response")
xPushGamPred <- data.frame(pdata,"displacement" = xPushGamPred)
pdata <- data.frame("xposition" = rep(seq(min(xPullDF$xposition),max(xPullDF$xposition),length.out=50),times=50),
                    "duration" = rep(seq(1000,max(xPullDF$duration),length.out=50),each=50))
xPullGamPred <- predict(xPullGam,pdata,type="response")
xPullGamPred <- data.frame(pdata,"displacement" = xPullGamPred)
pdata <- data.frame("yposition" = rep(seq(min(yPushDF$yposition),max(yPushDF$yposition),length.out=50),times=50),
                    "duration" = rep(seq(1000,max(yPushDF$duration),length.out=50),each=50))
yPushGamPred <- predict(yPushGam,pdata,type="response")
yPushGamPred <- data.frame(pdata,"displacement" = yPushGamPred)
pdata <- data.frame("yposition" = rep(seq(min(yPushDF$yposition),max(yPullDF$yposition),length.out=50),times=50),
                    "duration" = rep(seq(1000,max(yPullDF$duration),length.out=50),each=50))
yPullGamPred <- predict(yPullGam,pdata,type="response")
yPullGamPred <- data.frame(pdata,"displacement" = yPullGamPred)

#save gam predictions
xPushGamPred <- round(xPushGamPred,digits=1)
xPullGamPred <- round(xPullGamPred,digits=1)
yPushGamPred <- round(yPushGamPred,digits=1)
yPullGamPred <- round(yPullGamPred,digits=1)
write.table(xPushGamPred$position,"/home/cns/Desktop/Trodes/linux/xPushGamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPushGamPred$duration,"/home/cns/Desktop/Trodes/linux/xPushGamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPushGamPred$displacement,"/home/cns/Desktop/Trodes/linux/xPushGamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullGamPred$position,"/home/cns/Desktop/Trodes/linux/xPullGamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullGamPred$duration,"/home/cns/Desktop/Trodes/linux/xPullGamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullGamPred$displacement,"/home/cns/Desktop/Trodes/linux/xPullGamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushGamPred$position,"/home/cns/Desktop/Trodes/linux/yPushGamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushGamPred$duration,"/home/cns/Desktop/Trodes/linux/yPushGamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushGamPred$displacement,"/home/cns/Desktop/Trodes/linux/yPushGamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullGamPred$position,"/home/cns/Desktop/Trodes/linux/yPullGamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullGamPred$duration,"/home/cns/Desktop/Trodes/linux/yPullGamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullGamPred$displacement,"/home/cns/Desktop/Trodes/linux/yPullGamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")


#fit scams
xPushScam <- scam(displacement ~ s(xposition,duration,k=c(10,10),bs="tesmi2"),data=xPushDF)
xPullScam <- scam(displacement ~ s(xposition,duration,k=c(10,10),bs="tesmi2"),data=xPullDF)
yPushScam <- scam(displacement ~ s(yposition,duration,k=c(10,10),bs="tesmi2"),data=yPushDF)
yPullScam <- scam(displacement ~ s(yposition,duration,k=c(10,10),bs="tesmi2"),data=yPullDF)

#plot scams
plot(xPushScam,residuals = FALSE,scheme=1,main="xPush displacement")
plot(xPullScam,residuals = FALSE,scheme=1,main="xPull displacement")
plot(yPushScam,residuals = FALSE,scheme=1,main="yPush displacement")
plot(yPullScam,residuals = FALSE,scheme=1,main="yPull displacement")

#get scam predictions
pdata <- data.frame("xposition" = rep(seq(min(xPushDF$xposition),max(xPushDF$xposition),length.out=50),times=50),
                    "duration" = rep(seq(1250,20000.0,length.out=50),each=50))
xPushScamPred <- predict(xPushScam,pdata,type="response")
xPushScamPred[xPushScamPred<0] <- 0
xPushScamPred <- data.frame(pdata,"displacement" = xPushScamPred)
pdata <- data.frame("xposition" = rep(seq(min(xPullDF$xposition),max(xPullDF$xposition),length.out=50),times=50),
                    "duration" = rep(seq(2500,30000.0,length.out=50),each=50))
xPullScamPred <- predict(xPullScam,pdata,type="response")
xPullScamPred[xPullScamPred<0] <- 0
xPullScamPred <- data.frame(pdata,"displacement" = xPullScamPred)
pdata <- data.frame("yposition" = rep(seq(min(yPushDF$yposition),max(yPushDF$yposition),length.out=50),times=50),
                    "duration" = rep(seq(3000,30000.0,length.out=50),each=50))
yPushScamPred <- predict(yPushScam,pdata,type="response")
yPushScamPred[yPushScamPred<0] <- 0
yPushScamPred <- data.frame(pdata,"displacement" = yPushScamPred)
pdata <- data.frame("yposition" = rep(seq(min(yPullDF$yposition),max(yPullDF$yposition),length.out=50),times=50),
                    "duration" = rep(seq(3000,30000.0,length.out=50),each=50))
yPullScamPred <- predict(yPullScam,pdata,type="response")
yPullScamPred[yPullScamPred<0] <- 0
yPullScamPred <- data.frame(pdata,"displacement" = yPullScamPred)

#save scam predictions
xPushScamPred <- round(xPushScamPred,digits=1)
xPullScamPred <- round(xPullScamPred,digits=1)
yPushScamPred <- round(yPushScamPred,digits=1)
yPullScamPred <- round(yPullScamPred,digits=1)
write.table(xPushScamPred$xposition,"/home/cns/Desktop/Trodes/linux/xPushScamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPushScamPred$duration,"/home/cns/Desktop/Trodes/linux/xPushScamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPushScamPred$displacement,"/home/cns/Desktop/Trodes/linux/xPushScamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullScamPred$xposition,"/home/cns/Desktop/Trodes/linux/xPullScamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullScamPred$duration,"/home/cns/Desktop/Trodes/linux/xPullScamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullScamPred$displacement,"/home/cns/Desktop/Trodes/linux/xPullScamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushScamPred$yposition,"/home/cns/Desktop/Trodes/linux/yPushScamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushScamPred$duration,"/home/cns/Desktop/Trodes/linux/yPushScamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushScamPred$displacement,"/home/cns/Desktop/Trodes/linux/yPushScamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullScamPred$yposition,"/home/cns/Desktop/Trodes/linux/yPullScamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullScamPred$duration,"/home/cns/Desktop/Trodes/linux/yPullScamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullScamPred$displacement,"/home/cns/Desktop/Trodes/linux/yPullScamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")

#fit qgams
xPushQgam <- qgam(displacement ~ te(xposition,duration,k=c(10,10)),data=xPushDF,qu=0.85)
xPullQgam <- qgam(displacement ~ te(xposition,duration,k=c(10,10)),data=xPullDF,qu=0.85)
yPushQgam <- qgam(displacement ~ te(yposition,duration,k=c(10,10)),data=yPushDF,qu=0.85)
yPullQgam <- qgam(displacement ~ te(yposition,duration,k=c(10,10)),data=yPullDF,qu=0.85)

#plot qgams
plot(xPushQgam,residuals = FALSE,scheme=1)
plot(xPullQgam,residuals = FALSE,scheme=1)
plot(yPushQgam,residuals = FALSE,scheme=1)
plot(yPullQgam,residuals = FALSE,scheme=1)

#get qgam predictions
pdata <- data.frame("xposition" = rep(seq(min(xPushDF$xposition),max(xPushDF$xposition),length.out=50),times=50),
                    "duration" = rep(seq(1250.0,20000.0,length.out=50),each=50))
xPushQgamPred <- predict(xPushQgam,pdata,type="response")
xPushQgamPred[xPushQgamPred<0] <- 0
xPushQgamPred <- data.frame(pdata,"displacement" = xPushQgamPred)
pdata <- data.frame("xposition" = rep(seq(min(xPullDF$xposition),max(xPullDF$xposition),length.out=50),times=50),
                    "duration" = rep(seq(2000.0,30000.0,length.out=50),each=50))
xPullQgamPred <- predict(xPullQgam,pdata,type="response")
xPullQgamPred[xPullQgamPred<0] <- 0
xPullQgamPred <- data.frame(pdata,"displacement" = xPullQgamPred)
pdata <- data.frame("yposition" = rep(seq(min(yPushDF$yposition),max(yPushDF$yposition),length.out=50),times=50),
                    "duration" = rep(seq(3000.0,30000.0,length.out=50),each=50))
yPushQgamPred <- predict(yPushQgam,pdata,type="response")
yPushQgamPred[yPushQgamPred<0] <- 0
yPushQgamPred <- data.frame(pdata,"displacement" = yPushQgamPred)
pdata <- data.frame("yposition" = rep(seq(min(yPullDF$yposition),max(yPullDF$yposition),length.out=50),times=50),
                    "duration" = rep(seq(3000.0,30000.0,length.out=50),each=50))
yPullQgamPred <- predict(yPullQgam,pdata,type="response")
yPullQgamPred[yPullQgamPred<0] <- 0
yPullQgamPred <- data.frame(pdata,"displacement" = yPullQgamPred)

#save qgam predictions
xPushQgamPred <- round(xPushQgamPred,digits=1)
xPullQgamPred <- round(xPullQgamPred,digits=1)
yPushQgamPred <- round(yPushQgamPred,digits=1)
yPullQgamPred <- round(yPullQgamPred,digits=1)
write.table(xPushQgamPred$position,"/home/cns/Desktop/Trodes/linux/xPushQgamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPushQgamPred$duration,"/home/cns/Desktop/Trodes/linux/xPushQgamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPushQgamPred$displacement,"/home/cns/Desktop/Trodes/linux/xPushQgamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullQgamPred$position,"/home/cns/Desktop/Trodes/linux/xPullQgamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullQgamPred$duration,"/home/cns/Desktop/Trodes/linux/xPullQgamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(xPullQgamPred$displacement,"/home/cns/Desktop/Trodes/linux/xPullQgamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushQgamPred$position,"/home/cns/Desktop/Trodes/linux/yPushQgamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushQgamPred$duration,"/home/cns/Desktop/Trodes/linux/yPushQgamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPushQgamPred$displacement,"/home/cns/Desktop/Trodes/linux/yPushQgamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullQgamPred$position,"/home/cns/Desktop/Trodes/linux/yPullQgamPos.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullQgamPred$duration,"/home/cns/Desktop/Trodes/linux/yPullQgamDur.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(yPullQgamPred$displacement,"/home/cns/Desktop/Trodes/linux/yPullQgamDis.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
