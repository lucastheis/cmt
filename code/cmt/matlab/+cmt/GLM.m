classdef GLM < cmt.Trainable
    %GLM is an implementation of generalized linear models.
    %   It is given by
    %       $p(y \mid \mathbf{x}) = q(y \mid g(\mathbf{w}^\top \mathbf{x} + b)),$
    %   where $q$ is typically from the exponential family and $g$ is some nonlinearity (inverse link function) which has to be specified.
    %
    %    <a href="matlab: doc cmt.GLM">Function and Method Overview</a>
    %
    %   See also: cmt.Trainable, cmt.ConditionalDistribution

    properties
        % Linear filter, $w$.
        weights;
        % Bias term, $b$.
        bias;
    end

    methods
        function self = GLM(dimIn, varargin)
            self@cmt.Trainable(dimIn, varargin{:});
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
    methods (Static)
        function obj = loadobj(S)
            obj = cmt.GLM.mexLoad(S, @cmt.GLM, {'dimIn'});
        end
    end
end
