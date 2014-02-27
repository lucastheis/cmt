stimulus = randn(10, 1000);
spikes   = double(randn(1, 1000) > 0.7);



%% Train model
model = cmt.GLM(int32(10));

model.train(stimulus, spikes)