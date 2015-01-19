stimulus = randn(10, 1000);
spikes   = double(randn(1, 1000) > 0.7);


%% Test STM
model = cmt.STM(5, 5)

model.weights

model.train(stimulus, spikes, 'callback', @callback_test);

delete(model)

%% Test GLM
model = cmt.GLM(10)

model.bias

model.train(stimulus, spikes, struct('callback', @callback_test))

delete(model)

%% Test MCGSM
model = cmt.MCGSM(10)

model.predictors

model.train(stimulus, spikes)

labels = model.samplePosterior(stimulus, spikes);

samples = model.sample(stimulus, labels);

delete(model)
