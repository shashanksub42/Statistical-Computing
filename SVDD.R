data <- read.table('C:/Users/shash/Desktop/UCF/Fall 2018/Stat Computing 1/Final Project/pb2.txt')
data = data.frame(data)

#Features
X = as.matrix(data[1:31, 2:5])

#Labels
y = as.matrix(data[1:31, 1])

#Outliers
test = data[32:nrow(data), 2:5]

rbf_kernel <- function(x1,x2,sig){
  K<-exp(-(1/sig^2)*t(x1-x2)%*%(x1-x2))
  return(as.numeric(K))
}

library('quadprog')

svddtrain <- function(X, y, C, sig){
  Dm = matrix(0, nrow(X), nrow(X))
  X = as.matrix(X);y = as.vector(y)
  for (i in 1:nrow(X)){
    for (j in 1:nrow(X)){
      Dm[i, j] = 2*rbf_kernel(X[i,], X[j,], sig)
    }
  }
  Dm<-Dm+diag(nrow(X))*1e-12
  dv = matrix(0, nrow(X))
  for (i in 1:nrow(X)){
    dv[i] = rbf_kernel(as.numeric(X[i,]), as.numeric(X[i,]), sig)
  }
  meq = 1
  Am = rbind(t(matrix(1, nrow(X))), diag(nrow(X)))
  Am = rbind(Am, -diag(nrow(X)))
  
  bv = rbind(1, matrix(0, nrow(X)))
  bv = rbind(bv, matrix(-2, nrow(X)))
  
  alpha_org = solve.QP(Dm, dv, t(Am), meq = meq, bvec = bv)$solution
  alphaindx = which(alpha_org>1e-10, arr.ind = TRUE)
  alpha = alpha_org[alphaindx]
  nsv = length(alphaindx)
  
  Xv = X[alphaindx,]
  
  list(alpha=alpha, nsv=nsv, Xv=Xv, sig=sig)
}

sigma = c(1:100)
NSV = numeric(100)
for (i in 1:100){
  NSV[i] = svmtrain(X, y, C = 100, i)$nsv
  cat("SVs with sigma = ", i, ": ", NSV[i], "\n")
}
plot(sigma, NSV)



r_squared <- function(model){
  alpha = model$alpha
  b = model$b
  nsv = model$nsv
  Xv = model$Xv
  sig = model$sig
  r_squared = 0
  for (s in 1:nsv){
    sum1 = 0
    for (j in 1:nsv){
      sum1 = sum1 + alpha[j]*rbf_kernel(Xv[s,], Xv[j,], sig)
    }
    sum1 = 2*(sum1)
    sum2 = 0
    for (j in 1:nsv){
      for (l in 1:nsv){
        sum2 = sum2 + alpha[j]*alpha[l]*rbf_kernel(Xv[j,], Xv[l,], sig)
      }
    }
    r_squared = r_squared + rbf_kernel(Xv[s,], Xv[s,], sig) - sum1 + sum2
  }
  r_squared = (1/nsv)*r_squared
  return(r_squared)
  
}

model1 = svddtrain(X, y, 1, 57)

Rsq = as.numeric(r_squared(model1))

distance <- function(model, z, sig){
  alpha = model$alpha
  Xv = as.matrix(model$Xv)
  sum1 = 0
  for (j in 1:nrow(Xv)){
    sum1 = sum1 + alpha[j]*rbf_kernel(z, as.vector(Xv[j, ]), sig)
  }
  sum1 = 2*(sum1)
  sum2 = 0
  for (j in 1:nrow(Xv)){
    for (l in 1:nrow(Xv)){
      sum2 = sum2 + alpha[j]*alpha[l]*rbf_kernel(as.vector(Xv[j, ]), as.vector(Xv[l, ]), sig)
    }
  }
  dist = rbf_kernel(z, z, sig) - sum1 + sum2
  return(dist)
}

pred = matrix(0, nrow(test))
for (z in 1:nrow(test)){
  pred[z] = distance(model1, as.integer(test[z,]), 60)
  cat("Prediction value ", z, " = ", pred[z], "\n")
}

mean(pred > Rsq)