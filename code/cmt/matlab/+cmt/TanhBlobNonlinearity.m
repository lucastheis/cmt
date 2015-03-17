classdef TanhBlobNonlinearity
    properties
        numComponents = 3;
        epsilon = 1e-12;
    end

    methods
        function self = TanhBlobNonlinearity(numComponents, epsilon)
            if nargin > 0
                self.numComponents = numComponents;
            end
            if nargin > 0
                self.epsilon = epsilon;
            end
        end
    end

end

