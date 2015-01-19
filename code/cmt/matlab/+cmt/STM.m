classdef STM < cmt.Trainable
    %STM is an implementation of the spike-triggered mixture model.
	%   The conditional distribution defined by the model is
	%      $p(y \mid x, z) = q(y \mid g(f(x, z)))$
    %   where $y$ is a scalar, $\x \in R^N$, $\z \in R^M$,
    %   $q$ is a univariate distribution, $g$ is some nonlinearity, and
    %   $f(\x, \z) = \log \sum_k \exp\left( \lambda \left[ \sum_l \beta_{kl} (\u_l^\top \x)^2 + \w_k \x + a_k \right] \right) / \lambda + \v^\top \z.$
    %
    %   Examples:
    %       model = cmt.STM(10, 5);
    %
    %    <a href="matlab: doc cmt.STM">Function and Method Overview</a>
    %
    %   See also: cmt.Trainable, cmt.ConditionalDistribution

    properties (SetAccess = private)
        % Dimensionality of linear inputs.
        dimInLinear;
        % Dimensionality of nonlinear inputs.
        dimInNonlinear;
        % Numer of predictors.
        numComponents;
        % Number of features available to approximate input covariances.
        numFeatures;
    end

    properties
        % Controls the sharpness of the soft-maximum implemented by the log-sum-exp, $\lambda$.
        sharpness;
        % Weights relating features and mixture components, $\beta_{kl}$.
        weights;
        % Features used for capturing structure in inputs, $u_l$.
        features;
        % Parameters relating inputs and outputs, $w_k$.
        predictors;
        % Parameters relating inputs and outputs, $v$.
        linearPredictor;
        % Bias terms controlling strength of each mixture component, $a_k$.
        biases;

        % Distribution whose average value is determined by output of nonlinearity.
        %distribution
        % Nonlinearity applied to output of log-sum-exp, $g$.
        %nonlinearity
    end

    methods
        function self = STM(dimInNonlinear, varargin)
            %STM creates a new STM object.
            %   Parameters:
            %       dimInNonlinear - dimensionality of nonlinear portion of input
            %       dimInLinear (optional) - dimensionality of linear portion of input (default: 0)
            %       numComponents (optional) - number of components (default: 3)
            %       numFeatures (optional) - number of quadratic features (default: dimInNonlinear)
            %       nonlinearity (optional) - nonlinearity applied to log-sum-exp, g (default: LogisticFunction)
            %       distribution (optional) - distribution of outputs, q (default: Bernoulli)
            %   Return:
            %       a new STM object
            self@cmt.Trainable(dimInNonlinear, varargin{:});
        end

        % Constant properties
        function v = get.dimInLinear(self)
            v = self.mexEval('dimInLinear');
        end

        function v = get.dimInNonlinear(self)
            v = self.mexEval('dimInNonlinear');
        end

        function v = get.numComponents(self)
            v = self.mexEval('numComponents');
        end

        function v = get.numFeatures(self)
            v = self.mexEval('numFeatures');
        end

        % Nonconstant properties
        function set.sharpness(self, v)
            self.mexEval('setSharpness', v);
        end

        function v = get.sharpness(self)
            v = self.mexEval('sharpness');
        end


        function set.weights(self, v)
            self.mexEval('setWeights', v);
        end

        function v = get.weights(self)
            v = self.mexEval('weights');
        end


        function set.features(self, v)
            self.mexEval('setFeatures', v);
        end

        function v = get.features(self)
            v = self.mexEval('features');
        end


        function set.predictors(self, v)
            self.mexEval('setPredictors', v);
        end

        function v = get.predictors(self)
            v = self.mexEval('predictors');
        end


        function set.linearPredictor(self, v)
            self.mexEval('setLinearPredictor', v);
        end

        function v = get.linearPredictor(self)
            v = self.mexEval('linearPredictor');
        end


        function set.biases(self, v)
            self.mexEval('setBiases', v);
        end

        function v = get.biases(self)
            v = self.mexEval('biases');
        end


        % Methods
        function value = response(self, input, varargin)
            %RESPONSE computes the response of each component,
            %   The input can be the full input in one matrix or divided in linear and nonlinear part, that is, the combined dimension of dimIn().
            %   Parameters:
            %       input - full inputs (linear and non linear) stored in columns
            %   Returns:
            %       one response for each component and each input
            value = self.mexEval('response', input, varargin{:});
        end

        function value = nonlinearResponses(self, input)
            %NONLINEARRESPONSES computes the nonlinear response of each component,
            %   $\sum_l \beta_{kl} (u_l^\top x)^2 + w_k x + a_k.$$
            %   The input can be the full input or just the nonlinear part, that is, it can be of dimension dimIn() or of dimension dimInNonlinear().
            %   Parameters:
            %       input - inputs stored in columns
            %   Returns:
            %       one response for each component and each input
            value = self.mexEval('nonlinearResponses', input);
        end

        function value = linearResponses(self, input)
            %LINEARRESPONSES computes the linear portion of the intenal response, $v^\top z$.
            %   The input can be the full input or just the linear part, that is, it can be of dimension dimIn() or of dimension dimInLinear().
            %   Parameters:
            %       input - inputs stored in columns
            %   Returns:
            %       the output of the linear filter
            value = self.mexEval('linearResponses', input);
        end
    end
    methods (Static)
        function obj = loadobj(S)
            obj = cmt.MexInterface.mexLoad(S, @cmt.STM, {'dimInNonlinear', 'dimInLinear', 'numComponents', 'numFeatures'});
        end
    end
end
