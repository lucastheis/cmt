classdef (Abstract) ConditionalDistribution < cmt.MexInterface
    properties (SetAccess = private)
        % Dimensionality of inputs.
        dimIn;
        % Dimensionality of outputs.
        dimOut;
    end

    methods
        function self = ConditionalDistribution(varargin)
            self@cmt.MexInterface(varargin{:});
        end


        function v = get.dimIn(self)
            v = self.mexEval('dimIn');
        end

        function v = get.dimOut(self)
            v = self.mexEval('dimOut');
        end


        function value = sample(self, input)
            %SAMPLE generates outputs for given inputs.
            %   Parameters:
            %       input - inputs stored in columns
            %   Returns:
            %       sampled outputs
            value = self.mexEval('sample', input);
        end

        function value = predict(self, input)
            %PREDICT computes the expectation value of the output.
            %   Parameters:
            %       input - inputs stored in columns
            %   Returns:
            %       expected value of outputs
            value = self.mexEval('predict', input);
        end

        function value = logLikelihood(self, input, output)
            %LOGLIKELIHOOD computes the conditional log-likelihood for the given data points in nats.
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns
            %   Returns:
            %       log-likelihood of the model evaluated for each data point
            value = self.mexEval('logLikelihood', input, output);
        end

        function loglikelihood = evaluate(self, input, output)
            %EVALUATE computes the average negative conditional log-likelihood for data.
            %   Computed in bits per output component (smaller is better).
            %   If a preconditioner is specified, the data is transformed
            %   before computing the likelihood and the result is corrected
            %   for the Jacobian of the transformation. Note that the data
            %   should *not* already be transformed when specifying a
            %   preconditioner.
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns
            %       preconditioner - preconditioner that is used to transform the data
            %   Returns:
            %       performance in bits per component
            loglikelihood = self.mexEval('evaluate', input, output);
        end
    end
end
