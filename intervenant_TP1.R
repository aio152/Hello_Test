mu = -0.01/2
t1 = 0.05
N = c(seq(10, 1000,by=10))
z = matrix(rnorm(N*3), nrow = N, ncol = 3)

g = function (z1,z2,z3){
    return (1/ (1+t1 * exp(sig*z1 + mu)* (1 + t1*exp(sig*z1 + mu)*exp(sig*z2 +mu) *( 1 + t1*exp(sig*z1 + mu)*exp(sig*z2 +mu)*exp(sig*z3 +mu))))}
I=c()
for(i in N){
I[i/10] =1/i *sum((g(z[i,1],z[i,2],z[i,3])))
}
moy = 1/(1 + 0.05) * I
}
