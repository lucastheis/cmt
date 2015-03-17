classdef ExponentialFunction

    properties
        epsilon = 1e-12;
    end

    methods
        function self = ExponentialFunction(epsilon)
            if nargin > 0
                self.epsilon = epsilon;
            end
        end
    end

end

