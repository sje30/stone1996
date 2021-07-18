## Fitting a back prop network to the data.
## [2021-07-18 Sun]


require(nnet)

left <- scan("data/image1.txt", nlines=1)
right <- scan("data/image2.txt", nlines=1)
z <- scan("data/shifts.txt", nlines=1)

inputs <- matrix(0, 1000, 10)
outputs <- rep(0, 1000)
p <- 1

for (i in 1:1000) {
  inputs[i,1:5] <- left[p:(p+4)]
  inputs[i,6:10] <- right[p:(p+4)]
  outputs[i] <- z[p+2]
  p <- p + 7
}

## Note the network has 121 weights rather than 120 as before -- the output unit
## has a bias unit too.

p = nnet(inputs, outputs, size=10, linout=T)

png(file="fitted_bp.png")
plot(p$fitted)
dev.off()
