classdef LogisticFunction

    properties
        epsilon = 1e-12;
    end

    methods
        function self = LogisticFunction(epsilon)
            if nargin > 0
                self.epsilon = epsilon;
            end
        end
    end

end

