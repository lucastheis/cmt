stimulus = randn(10, 1000);
spikes   = double(randn(1, 1000) > 0.7);



%% Train model
model = cmt.STM(int32(10), int32(0))

model.weights

model.train(stimulus, spikes);

delete(model)