library(mgcv)
library(scam)
library(qgam)
library(MASS)
library(rgl)
library(mgcViz)
Sys.setenv(LIBGL_ALWAYS_SOFTWARE=1)

#load data
calibrationDF <- read.csv(file="/home/cns/Desktop/Trodes/linux/calibrationDF.csv", header=TRUE, sep=",")

#preprocess data (convert to bits, microseconds, etc.)
calibrationDF$x_displacement <- calibrationDF$x_displacement/5*1023
calibrationDF$y_displacement <- calibrationDF$y_displacement/5*1023
calibrationDF$z_displacement <- calibrationDF$z_displacement/5*1023
# calibrationDF$x_displacement[calibrationDF$x_duration>0 && calibrationDF$x_displacement<0] <- 0
# calibrationDF$x_displacement[calibrationDF$x_duration<0 && calibrationDF$x_displacement>0] <- 0
# calibrationDF$y_displacement[calibrationDF$y_duration>0 && calibrationDF$y_displacement<0] <- 0
# calibrationDF$y_displacement[calibrationDF$y_duration<0 && calibrationDF$y_displacement>0] <- 0
# calibrationDF$z_displacement[calibrationDF$z_duration>0 && calibrationDF$z_displacement<0] <- 0
# calibrationDF$z_displacement[calibrationDF$z_duration<0 && calibrationDF$z_displacement>0] <- 0
calibrationDF$x_position <- calibrationDF$x_position/5*1023
calibrationDF$y_position <- calibrationDF$y_position/5*1023
calibrationDF$z_position <- calibrationDF$z_position/5*1023
calibrationDF$x_duration <- calibrationDF$x_duration*1000000
calibrationDF$y_duration <- calibrationDF$y_duration*1000000
calibrationDF$z_duration <- calibrationDF$z_duration*1000000

#fit scams
xScam <- gam(x_displacement ~ te(x_position,x_duration),data=calibrationDF)
yScam <- gam(y_displacement ~ te(y_position,y_duration),data=calibrationDF)
zScam <- gam(z_displacement ~ te(z_position,z_duration),data=calibrationDF)

xViz <- getViz(xScam)
yViz <- getViz(yScam)
zViz <- getViz(zScam)
plotRGL(sm(zViz,1))
# xScam <- bam(x_displacement ~ te(x_position,x_duration),data=calibrationDF)
# yScam <- bam(y_displacement ~ te(y_position,y_duration),data=calibrationDF)
# zScam <- bam(z_displacement ~ te(z_position,z_duration),data=calibrationDF)

#plot scams
plot(xScam,residuals = FALSE,scheme=1,main="x displacement")
plot(yScam,residuals = FALSE,scheme=1,main="y displacement")
plot(zScam,residuals = FALSE,scheme=1,main="z displacement")

#get scam predictions
npos <- 50
ndur <- 5000
ndis <- 50
pos <- seq(0,1023,length.out=npos)
pos = round(pos,digits = 0)
want <- grep("duration",colnames(calibrationDF))
dur <- seq(0,max(abs(calibrationDF[,want])),length.out = ndur)
dis <- seq(0,1023,length.out = ndis)
dis = round(dis,digits = 0)
xPushData <- data.frame("x_position" = rep(pos,times=ndur),
                        "x_duration" = rep(dur,each=npos))
xPullData <- data.frame("x_position" = rep(pos,times=ndur),
                        "x_duration" = -rep(dur,each=npos))
yPushData <- data.frame("y_position" = rep(pos,times=ndur),
                        "y_duration" = rep(dur,each=npos))
yPullData <- data.frame("y_position" = rep(pos,times=ndur),
                        "y_duration" = -rep(dur,each=npos))
zPushData <- data.frame("z_position" = rep(pos,times=ndur),
                        "z_duration" = rep(dur,each=npos))
zPullData <- data.frame("z_position" = rep(pos,times=ndur),
                        "z_duration" = -rep(dur,each=npos))
xPushPred <- predict(xScam,xPushData,se=TRUE)
xPullPred <- predict(xScam,xPullData,se=TRUE)
yPushPred <- predict(yScam,yPushData,se=TRUE)
yPullPred <- predict(yScam,yPullData,se=TRUE)
zPushPred <- predict(zScam,zPushData,se=TRUE)
zPullPred <- predict(zScam,zPullData,se=TRUE)

#get prediction probabilities
xPushProb <- array(0,c(npos,ndur,ndis))
xPullProb <- array(0,c(npos,ndur,ndis))
yPushProb <- array(0,c(npos,ndur,ndis))
yPullProb <- array(0,c(npos,ndur,ndis))
zPushProb <- array(0,c(npos,ndur,ndis))
zPullProb <- array(0,c(npos,ndur,ndis))
for (p in 1:npos){
  for (d in 1:ndur){
    i <- (d-1)*npos + p
    xPushProb[p,d,] <- dnorm(dis,xPushPred[[1]][i],xPushPred[[2]][i])
    xPullProb[p,d,] <- dnorm(-dis,xPullPred[[1]][i],xPullPred[[2]][i])
    yPushProb[p,d,] <- dnorm(dis,yPushPred[[1]][i],yPushPred[[2]][i])
    yPullProb[p,d,] <- dnorm(-dis,yPullPred[[1]][i],yPullPred[[2]][i])
    zPushProb[p,d,] <- dnorm(dis,zPushPred[[1]][i],zPushPred[[2]][i])
    zPullProb[p,d,] <- dnorm(-dis,zPullPred[[1]][i],zPullPred[[2]][i])
  }
}

#get durations with max probability for each position/displacement 
xPushDur <- rep(0,times=ndis*npos)
xPullDur <- rep(0,times=ndis*npos)
yPushDur <- rep(0,times=ndis*npos)
yPullDur <- rep(0,times=ndis*npos)
zPushDur <- rep(0,times=ndis*npos)
zPullDur <- rep(0,times=ndis*npos)
xPushDurP <- rep(0,times=ndis*npos)
xPullDurP <- rep(0,times=ndis*npos)
yPushDurP <- rep(0,times=ndis*npos)
yPullDurP <- rep(0,times=ndis*npos)
zPushDurP <- rep(0,times=ndis*npos)
zPullDurP <- rep(0,times=ndis*npos)
for (d in 1:ndis){
  for (p in 1:npos){
    i <- (d-1)*npos + p
    xPushDur[i] <- dur[which.max(xPushProb[p,,d])]
    xPullDur[i] <- dur[which.max(xPullProb[p,,d])]
    yPushDur[i] <- dur[which.max(yPushProb[p,,d])]
    yPullDur[i] <- dur[which.max(yPullProb[p,,d])]
    zPushDur[i] <- dur[which.max(zPushProb[p,,d])]
    zPullDur[i] <- dur[which.max(zPullProb[p,,d])]
    
    xPushDurP[i] <- max(xPushProb[p,,d])
    xPullDurP[i] <- max(xPullProb[p,,d])
    yPushDurP[i] <- max(yPushProb[p,,d])
    yPullDurP[i] <- max(yPullProb[p,,d])
    zPushDurP[i] <- max(zPushProb[p,,d])
    zPullDurP[i] <- max(zPullProb[p,,d])
  }
}

# #get durations with expected displacement for a given position
# xPushDur <- rep(0,times=ndis*npos)
# xPullDur <- rep(0,times=ndis*npos)
# yPushDur <- rep(0,times=ndis*npos)
# yPullDur <- rep(0,times=ndis*npos)
# zPushDur <- rep(0,times=ndis*npos)
# zPullDur <- rep(0,times=ndis*npos)
# xPushDurP <- rep(0,times=ndis*npos)
# xPullDurP <- rep(0,times=ndis*npos)
# yPushDurP <- rep(0,times=ndis*npos)
# yPullDurP <- rep(0,times=ndis*npos)
# zPushDurP <- rep(0,times=ndis*npos)
# zPullDurP <- rep(0,times=ndis*npos)
# for (d in 1:ndis){
#   for (p in 1:npos){
#     i <- (d-1)*npos + p
#     xPushDur[i] <- min(xPushData$x_duration[xPushPred[[1]]>=dis[d] & xPushData$x_position==pos[p]])
#     xPullDur[i] <- min(-xPullData$x_duration[-xPullPred[[1]]>=dis[d] & xPullData$x_position==pos[p]])
#     yPushDur[i] <- min(yPushData$y_duration[yPushPred[[1]]>=dis[d] & yPushData$y_position==pos[p]])
#     yPullDur[i] <- min(-yPullData$y_duration[-yPullPred[[1]]>=dis[d] & yPullData$y_position==pos[p]])
#     zPushDur[i] <- min(zPushData$z_duration[zPushPred[[1]]>=dis[d] & zPushData$z_position==pos[p]])
#     zPullDur[i] <- min(-zPullData$z_duration[-zPullPred[[1]]>=dis[d] & zPullData$z_position==pos[p]])
# 
#   }
# }
# xPushDur[xPushDur>max(max(abs(calibrationDF[,want])))] <- max(abs(calibrationDF[,want]))
# xPullDur[xPullDur>max(max(abs(calibrationDF[,want])))] <- max(abs(calibrationDF[,want]))
# yPushDur[yPushDur>max(max(abs(calibrationDF[,want])))] <- max(abs(calibrationDF[,want]))
# yPullDur[yPullDur>max(max(abs(calibrationDF[,want])))] <- max(abs(calibrationDF[,want]))
# zPushDur[zPushDur>max(max(abs(calibrationDF[,want])))] <- max(abs(calibrationDF[,want]))
# zPullDur[zPullDur>max(max(abs(calibrationDF[,want])))] <- max(abs(calibrationDF[,want]))

#set low certainty actions to nearest high certainty neighbor values
probThresh <- 0.0025
idx <- seq(1,npos*ndis)
idxLowP <- which(xPushDurP < probThresh)
for (i in idxLowP){
  dLowP <- floor(i/npos) + 1
  pLowP <- i%%npos
  idxHighP <- idx[-idxLowP]
  dHighP <- floor(idxHighP/npos) + 1
  pHighP <- idxHighP%%npos
  nearestIdx <- idxHighP[which.min((dHighP-dLowP)^2 + (pHighP-pLowP)^2)]
  xPushDur[i] <- xPushDur[nearestIdx]
}
idxLowP <- which(xPullDurP < probThresh)
for (i in idxLowP){
  dLowP <- floor(i/npos) + 1
  pLowP <- i%%npos
  idxHighP <- idx[-idxLowP]
  dHighP <- floor(idxHighP/npos) + 1
  pHighP <- idxHighP%%npos
  nearestIdx <- idxHighP[which.min((dHighP-dLowP)^2 + (pHighP-pLowP)^2)]
  xPullDur[i] <- xPullDur[nearestIdx]
}
idxLowP <- which(yPushDurP < probThresh)
for (i in idxLowP){
  dLowP <- floor(i/npos) + 1
  pLowP <- i%%npos
  idxHighP <- idx[-idxLowP]
  dHighP <- floor(idxHighP/npos) + 1
  pHighP <- idxHighP%%npos
  nearestIdx <- idxHighP[which.min((dHighP-dLowP)^2 + (pHighP-pLowP)^2)]
  yPushDur[i] <- yPushDur[nearestIdx]
}
idxLowP <- which(yPullDurP < probThresh)
for (i in idxLowP){
  dLowP <- floor(i/npos) + 1
  pLowP <- i%%npos
  idxHighP <- idx[-idxLowP]
  dHighP <- floor(idxHighP/npos) + 1
  pHighP <- idxHighP%%npos
  nearestIdx <- idxHighP[which.min((dHighP-dLowP)^2 + (pHighP-pLowP)^2)]
  yPullDur[i] <- yPullDur[nearestIdx]
}
idxLowP <- which(zPushDurP < probThresh)
for (i in idxLowP){
  dLowP <- floor(i/npos) + 1
  pLowP <- i%%npos
  idxHighP <- idx[-idxLowP]
  dHighP <- floor(idxHighP/npos) + 1
  pHighP <- idxHighP%%npos
  nearestIdx <- idxHighP[which.min((dHighP-dLowP)^2 + (pHighP-pLowP)^2)]
  zPushDur[i] <- zPushDur[nearestIdx]
}
idxLowP <- which(zPullDurP < probThresh)
for (i in idxLowP){
  dLowP <- floor(i/npos) + 1
  pLowP <- i%%npos
  idxHighP <- idx[-idxLowP]
  dHighP <- floor(idxHighP/npos) + 1
  pHighP <- idxHighP%%npos
  nearestIdx <- idxHighP[which.min((dHighP-dLowP)^2 + (pHighP-pLowP)^2)]
  zPullDur[i] <- zPullDur[nearestIdx]
}
xPushDur[xPushDur<3000] <- 3000
xPullDur[xPullDur<5000] <- 5000
yPushDur[yPushDur<5000] <- 5000
yPullDur[yPullDur<3500] <- 3500
zPushDur[zPushDur<6000] <- 6000
zPullDur[zPullDur<11000] <- 13000

#combine into data frames
fullPred <- data.frame("displacement" = rep(dis,each=npos),
                       "position" = rep(pos,times=ndis),
                       "xPushDuration" = xPushDur,
                       "xPullDuration" = xPullDur,
                       "yPushDuration" = yPushDur,
                       "yPullDuration" = yPullDur,
                       "zPushDuration" = zPushDur,
                       "zPullDuration" = zPullDur)
fullPred <- round(fullPred,digits=1)

write.table(fullPred,"/home/cns/Desktop/calibrationFile.txt",sep=",")
write.table(fullPred$position,"/home/cns/Desktop/Trodes/linux/position.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(fullPred$displacement,"/home/cns/Desktop/Trodes/linux/displacement.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(fullPred$xPushDuration,"/home/cns/Desktop/Trodes/linux/xPushDuration.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(fullPred$xPullDuration,"/home/cns/Desktop/Trodes/linux/xPullDuration.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(fullPred$yPushDuration,"/home/cns/Desktop/Trodes/linux/yPushDuration.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(fullPred$yPullDuration,"/home/cns/Desktop/Trodes/linux/yPullDuration.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(fullPred$zPushDuration,"/home/cns/Desktop/Trodes/linux/zPushDuration.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
write.table(fullPred$zPullDuration,"/home/cns/Desktop/Trodes/linux/zPullDuration.txt",sep=",",row.names=FALSE,col.names=FALSE,eol = ",")
