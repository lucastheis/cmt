classdef STM < cmt.Trainable
    properties (Dependent)
        sharpness;
        weights;
        features;
        predictors;
        linearPredictor;
    end
  
    methods
        function self = STM(dimInNonlinear, dimInLinear, varargin)
            self@cmt.Trainable(dimInNonlinear, dimInLinear, varargin{:});
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
end
