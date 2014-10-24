classdef STM < cmt.Trainable
    %STM is an implementation of the spike-triggered mixture model.
	%   The conditional distribution defined by the model is
    %   <latex>
	%   $p(y \mid \mathbf{x}, \mathbf{z}) = q(y \mid g(f(\mathbf{x}, \mathbf{z})))$
	%   <\latex>
    %   where $y$ is a scalar, $\\mathbf{x} \\in \\mathbb{R}^N$, $\\mathbf{z} \\in \\mathbb{R}^M$,
    %   $q$ is a univariate distribution, $g$ is some nonlinearity, and
    %   $f(\\mathbf{x}, \\mathbf{z}) = \\log \\sum_k \\exp\\left( \\lambda \\left[ \\sum_l \\beta_{kl} (\\mathbf{u}_l^\\top \\mathbf{x})^2 + \\mathbf{w}_k \\mathbf{x} + a_k \\right] \\right) / \\lambda + \\mathbf{v}^\\top \\mathbf{z}.$$\n"

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
        distribution
        % Nonlinearity applied to output of log-sum-exp, $g$.
        nonlinearity
    end

    methods
        function self = STM(varargin)
            % dimNonlinear (int) - dimensionality of nonlinear portion of input
            % dimLinear (int) - dimensionality of linear portion of input (default: 0)
            % numComponents (int) - number of components (default: 3)
            % numFeatures (int) - number of quadratic features
            % nonlinearity (Nonlinearity/type) - nonlinearity applied to log-sum-exp, g (default: LogisticFunction)
            % distribution (UnivariateDistribution/type) - distribution of outputs, q (default: Bernoulli)
            self@cmt.Trainable(varargin{:});
        end


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

        % Methods
        function value = response(self, input, varargin)
            value = self.mexEval('response', input, varargin{:});
        end

        function value = nonlinearResponses(self, input)
            value = self.mexEval('nonlinearResponses', input);
        end

        function value = linearResponses(self, input)
            value = self.mexEval('linearResponses', input);
        end
    end
    methods (Static)
        function obj = loadobj(S)
            obj = cmt.MexInterface.mexLoad(S, @cmt.STM, {'dimInNonlinear', 'dimInLinear', 'numComponents', 'numFeatures'});
        end
    end
end
