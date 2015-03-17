classdef Binomial

    properties
        n = 10;
        p = 0.5;
    end

    methods
        function self = Binomial(n, p)
            if nargin > 0
                self.n = n;
            end
            if nargin > 1
                self.p = p;
            end
        end
    end

end

