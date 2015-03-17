classdef HistogramNonlinearity

    properties
        inputs;
        outputs;
        numBins;
        epsilon = 1e-12;
    end

    methods
        function self = HistogramNonlinearity(inputs, outputs, numBins, epsilon)
            self.inputs = inputs;
            self.outputs = outputs;
            self.numBins = numBins;

            if nargin > 3
                self.epsilon = epsilon;
            end
        end
    end

end

