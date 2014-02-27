classdef (Abstract) Trainable < cmt.ConditionalDistribution
    methods
        function self = Trainable(varargin)
            self@cmt.ConditionalDistribution(varargin{:});
        end
        
        function initialize(self, input, output)
            self.mexEval('initialize', input, output);
        end
        
        function bool = train(self, input, output, varargin)
            bool = self.mexEval('train', input, output, varargin{:});
        end
        
        function value = checkGradient(self, input, output, varargin)
            value = self.mexEval('checkGradient', input, output, varargin{:});
        end
        
        function value = checkPerformance(self, input, output, varargin)
            value = self.mexEval('checkPerformance', input, output, varargin{:});
        end
        
        function matrix = fisherInformation(self, input, output)
            matrix = self.mexEval('fisherInformation', input, output);
        end
    end
end