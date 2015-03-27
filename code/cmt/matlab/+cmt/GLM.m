classdef GLM < cmt.Trainable
    %GLM is an implementation of generalized linear models.
    %   It is given by
    %       $p(y \mid \mathbf{x}) = q(y \mid g(\mathbf{w}^\top \mathbf{x} + b)),$
    %   where $q$ is typically from the exponential family and $g$ is some nonlinearity (inverse link function) which has to be specified.
    %
    %    <a href="matlab: doc cmt.GLM">Function and Method Overview</a>
    %
    %   See also: cmt.Trainable, cmt.ConditionalDistribution

    properties (SetAccess = private)
        % Nonlinearity applied to output of log-sum-exp, $g$.
        nonlinearity = cmt.LogisticFunction;
        % Distribution whose average value is determined by output of nonlinearity.
        distribution = cmt.Bernoulli;
    end

    properties
        % Linear filter, $w$.
        weights;
        % Bias term, $b$.
        bias;
    end

    methods
        function self = GLM(dimIn, varargin)
            self@cmt.Trainable(dimIn, varargin{:});

            % These can not be read from the C++ object, so Matlab has to keep track of them.
            if nargin > 1
                self.nonlinearity = varargin{1};
            end
            if nargin > 2
                self.distribution = varargin{2};
            end
        end


        function set.weights(self, v)
            self.mexEval('setWeights', v);
        end

        function v = get.weights(self)
            v = self.mexEval('weights');
        end


        function set.bias(self, v)
            self.mexEval('setBias', v);
        end

        function v = get.bias(self)
            v = self.mexEval('bias');
        end
    end

    properties (Constant, Hidden)
        constructor_arguments = {'dimIn', 'nonlinearity', 'distribution'};
        dependend_properties = {'dimOut'};
    end

    methods (Static)
        function obj = loadobj(S)
            obj = cmt.GLM.mexLoad(S, @cmt.GLM, cmt.GLM.constructor_arguments, ...
                                               cmt.GLM.dependend_properties);
        end
    end
end
