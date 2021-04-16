module Lenet5
using Knet, MLDatasets, IterTools
print("sss")


## Conv Layers
struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b))
Conv(w1,w2,cx,cy,f=relu) = Conv(param(w1,w2,cx,cy), param0(1,1,cy,1), f);


## Dense Layers
struct Dense; w; b; f; end
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b)
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f);

#  Chain of layers:
struct Chain; layers; Chain(args...)=new(args); end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)

# Load MNIST data
xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10
dtrn = minibatch(xtrn, ytrn, 100; xsize=(size(xtrn,1),size(xtrn,2),1,:))
dtst = minibatch(xtst, ytst, 100; xsize=(size(xtst,1),size(xtst,2),1,:));

# Train and test LeNet (about 30 secs on a gpu to reach 99% accuracy)
LeNet = Chain(Conv(5,5,1,20), Conv(5,5,20,50), Dense(800,500), Dense(500,10,identity))
progress!(adam(LeNet, ncycle(dtrn,10)))
accuracy(LeNet, dtst)

end