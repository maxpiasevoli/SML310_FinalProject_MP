library(rstan)
library(ggplot2)
library(reshape2)

setwd("C:/Users/maxpi/Documents/SML310/FinalProject")

# Make Synthetic distribution
n = 5000
Z = runif(n, -1, 1)
h1 = rnorm(n, plogis(Z), 1)
h2 = rnorm(n, plogis(-Z), 1)

X1 = rnorm(n, h1 + h2, 3)
X2 = rnorm(n, 2*h1 - 3*h2, 5)
X3 = rnorm(h1, 7)
X4 = rnorm(h2, 9)
X5 = rnorm(h1 * h2, 9)
X6 = rnorm(h1/h2, 9)

X = cbind(X1, X2, X3, X4, X5, X6)
#write.csv(X, row.names=FALSE, file='synDist.csv')

# behavioral learning experiment hierarchical models

# read-in data 
y1 <- as.matrix (read.table ("./data/dogs.dat"), nrows=30, ncol=25)
y <- ifelse (y1[,]=="S",1,0)
n.dogs <- nrow(y)
n.trials <- ncol(y)

## fit model 1
dataList.1 <- list(n_dogs=n.dogs,n_trials=n.trials,y=y)
dogs.sf1 <- stan(file='dogs.stan', data=dataList.1, iter=1000, chains=4) # model with intercept
print(dogs.sf1, pars = c("beta","lp__"))
post1 <- extract(dogs.sf1)
beta1 <- colMeans(post1$beta)

## generate samples for model 1
n.sims <- 1
y.rep.m1 <- array (NA, c(n.sims, n.dogs, n.trials))
for (j in 1:n.dogs){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  for (t in 1:n.trials){  
    p.rep <- plogis (beta1[1] + beta1[2]*n.avoid.rep + beta1[3]*n.shock.rep)
    y.rep.m1[,j,t] <- rbinom (n.sims, 1, p.rep)
    n.avoid.rep <- n.avoid.rep + 1 - y.rep.m1[,j,t] 
    n.shock.rep <- n.shock.rep + y.rep.m1[,j,t] 
  }
}
y.rep.m1[1,,]
#write.csv(y.rep.m1[1,,], file='./bl_m1.csv')

dogs.sf2 <- stan(file='dogs_no_int.stan', data=dataList.1, iter=1000, chains=4) # model with intercept
print(dogs.sf2, pars = c("beta","lp__"))
post2 <- extract(dogs.sf2)
beta2 <- colMeans(post2$beta)

n.sims <- 1
y.rep.m2 <- array (NA, c(n.sims, n.dogs, n.trials))
for (j in 1:n.dogs){
  n.avoid.rep <- rep (0, n.sims)
  n.shock.rep <- rep (0, n.sims)
  for (t in 1:n.trials){  
    p.rep <- exp(beta2[1]*n.avoid.rep + beta2[2]*n.shock.rep)
    y.rep.m2[,j,t] <- rbinom (n.sims, 1, p.rep)
    n.avoid.rep <- n.avoid.rep + 1 - y.rep.m2[,j,t] 
    n.shock.rep <- n.shock.rep + y.rep.m2[,j,t] 
  }
}
y.rep.m2[1,,]
#write.csv(y.rep.m2[1,,], file='./bl_m2.csv')

# stop and frisk models
n.ep <- as.data.frame(read.csv("./data/2014_arrests_ones.csv", sep=","))
y.ep <- as.data.frame(read.csv("./data/20152016_stops.csv", sep=","))
n.ep.z <- as.data.frame(read.csv("./data/2014_arrests_zeros.csv", sep=","))
#n.ep <- as.data.frame(read.csv("2014_arrests_all_crimes.csv", sep=","))
#y.ep <- as.data.frame(read.csv("20152016_stops_all_crimes.csv", sep=","))
#n.ep.z <- as.data.frame(read.csv("2014_arrests_zeros_all_crimes.csv", sep=","))


n.ep.lt10 <- subset(n.ep, Ethnic_Comp_Cat==0, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
n.ep.1040 <- subset(n.ep, Ethnic_Comp_Cat==1, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
n.ep.gt40 <- subset(n.ep, Ethnic_Comp_Cat==2, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
n.ep.z.lt10 <- subset(n.ep.z, Ethnic_Comp_Cat==0, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
n.ep.z.1040 <- subset(n.ep.z, Ethnic_Comp_Cat==1, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
n.ep.z.gt40 <- subset(n.ep.z, Ethnic_Comp_Cat==2, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
y.ep.lt10 <- subset(y.ep, Ethnic_Comp_Cat==0, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
y.ep.1040 <- subset(y.ep, Ethnic_Comp_Cat==1, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))
y.ep.gt40 <- subset(y.ep, Ethnic_Comp_Cat==2, select=c("Precinct", "Race", "Race_Int", "Eth_Pop_In_Precinct", "Occurrences"))

# lt10 hm fit for model 3
numIters = 1000
nChains = 4
current.n.ep = n.ep.lt10 
current.y.ep = y.ep.lt10
sample.index <- nrow(current.n.ep)
dataList.2 <- list(sample_index=sample.index, n=current.n.ep$Occurrences, y=current.y.ep$Occurrences, sample_race=current.y.ep$Race_Int)
saf.m3 <- stan(file='sf_model3.stan', 
               data=dataList.2,
               chains = nChains,
               iter = numIters) 
print(saf.m3, pars = c("beta","lp__"))

# extract parameters 
post3 <- extract(saf.m3)
alpha_m3 <- colMeans(post3$alpha)
beta_m3 <- colMeans(post3$beta)
mu_m3 <- colMeans(post3$mu)
epsilon_m3 <- colMeans(post3$epsilon)

# generate samples from model 3
lambdas_m3 = c(1:sample.index)
for (i in 1:sample.index) {
  lambdas_m3[i] = (15/12) * current.n.ep$Occurrences[i] * exp(mu_m3 + alpha_m3[current.n.ep$Race_Int[i]] + beta_m3[ceiling(i/3)] + epsilon_m3[i])
}
samples_m3 = rpois(sample.index, lambdas_m3)
current.y.ep$Y_Pred_M3 = samples_m3
alpha_m3


# lt10 hm fit for model 4
saf.m4 <- stan(file='sf_model4.stan', 
               data=dataList.2, 
               chains = nChains,
               iter = numIters) 
print(saf.m4, pars = c("beta","lp__"))

# extract parameters
post4 <- extract(saf.m4)
alpha_m4 <- colMeans(post4$alpha)
gamma <- colMeans(post4$gamma)
beta_m4 <- colMeans(post4$beta)
mu_m4 <- colMeans(post4$mu)
epsilon_m4 <- colMeans(post4$epsilon)

# generate samples from model 4
lambdas_m4 = c(1:sample.index)
for (i in 1:sample.index) {
  lambdas_m4[i] = (15/12) * exp(gamma * log(current.n.ep$Occurrences[i]) + mu_m4 + alpha_m4[current.n.ep$Race_Int[i]] + beta_m4[ceiling(i/3)] + epsilon_m4[i])
}
samples_m4 = rpois(sample.index, lambdas_m4)
current.y.ep$Y_Pred_M4 = samples_m4
alpha_m4


# lt10 hm fit for model 5
dataList.3 <- list(sample_index=sample.index, n=current.n.ep$Occurrences, y=current.y.ep$Occurrences, sample_race=current.y.ep$Race_Int, eth_pop=current.y.ep$Eth_Pop_In_Precinct)
saf.m5 <- stan(file='sf_model5.stan', 
               data=dataList.3, 
               iter=numIters, 
               chains=nChains) 
print(saf.m5, pars = c("beta","lp__"))

# extract parameters
post5 <- extract(saf.m5)
alpha_m5 <- colMeans(post5$alpha)
theta <- colMeans(post5$theta)
beta_m5 <- colMeans(post5$beta)
mu_m5 <- colMeans(post5$mu)
epsilon_m5 <- colMeans(post5$epsilon)

# generate samples from model 5
lambdas_m5 = c(1:sample.index)
for (i in 1:sample.index) {
  lambdas_m5[i] = (15/12) * theta[i] * exp(mu_m5 + alpha_m5[current.n.ep$Race_Int[i]] + beta_m5[ceiling(i/3)] + epsilon_m5[i])
}
samples_m5 = rpois(sample.index, lambdas_m5)
current.y.ep$Y_Pred_M5 = samples_m5
alpha_m5

#write.csv(current.y.ep, file='./sf_all_models_w_reps.csv')

# combine ethnic populations with predicted stops for wgan
eth_pops = matrix(n.ep.lt10$Eth_Pop_In_Precinct, nrow=30, byrow=TRUE)
recorded_stops = matrix(current.y.ep$Occurrences, nrow=30, byrow=TRUE)
model_3_stops = matrix(current.y.ep$Y_Pred_M3, nrow=30, byrow=TRUE)
model_4_stops = matrix(current.y.ep$Y_Pred_M4, nrow=30, byrow=TRUE)
model_5_stops = matrix(current.y.ep$Y_Pred_M5, nrow=30, byrow=TRUE)
eth_total = 1:nrow(eth_pops)
for (i in 1:nrow(eth_pops)) {
  eth_total[i] = eth_pops[i,1] + eth_pops[i,2] + eth_pops[i,3]
}

pops_and_preds_m3 = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               pred_white = model_3_stops[,1],
                               pred_black = model_3_stops[,2],
                               pred_hisp = model_3_stops[,3])
pops_and_preds_m4 = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               pred_white = model_4_stops[,1],
                               pred_black = model_4_stops[,2],
                               pred_hisp = model_4_stops[,3])
pops_and_preds_m5 = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               pred_white = model_5_stops[,1],
                               pred_black = model_5_stops[,2],
                               pred_hisp = model_5_stops[,3])
pops_and_recorded = data.frame(white_pop = eth_pops[,1],
                               black_pop = eth_pops[,2],
                               hisp_pop = eth_pops[,3],
                               pop_total = eth_total,
                               stops_white = recorded_stops[,1],
                               stops_black = recorded_stops[,2],
                               stops_hisp = recorded_stops[,3])
write.csv(pops_and_preds_m3, file='./pops_and_preds_m3.csv')
write.csv(pops_and_preds_m4, file='./pops_and_preds_m4.csv')
write.csv(pops_and_preds_m5, file='./pops_and_preds_m5.csv')
write.csv(pops_and_recorded, file='./pops_and_recorded.csv')
    
