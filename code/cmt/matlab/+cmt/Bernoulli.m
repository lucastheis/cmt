classdef Bernoulli

    properties
        prob = 0.5;
    end

    methods
        function self = Bernoulli(prob)
            if nargin > 0
                self.prob = prob;
            end
        end
    end

end

