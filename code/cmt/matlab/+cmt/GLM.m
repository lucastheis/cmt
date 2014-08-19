classdef GLM < cmt.Trainable
    properties
        weights;
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
            obj = cmt.GLM.mexLoad(S, @cmt.GLM);
        end
    end
end
