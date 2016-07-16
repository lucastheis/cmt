classdef Poisson

    properties
        lambda = 1;
    end

    methods
        function self = Poisson(lambda)
            if nargin > 0
                self.lambda = lambda;
            end
        end
    end

end

