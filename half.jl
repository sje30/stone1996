## Check the half-life calculations.
h = 32  ## half-life
l = 2^(-1/h) ## equivalent lambda


global z = 1.0
o = [z]

for i = 1:100
    global z = l*z + (1-l)*0
    push!(o, z)
end

plot(o)
vline!([h])
hline!([0.5], col="red")
