classdef BlobNonlinearity

    properties
        numComponents = 3;
        epsilon = 1e-12;
    end

    methods
        function self = BlobNonlinearity(numComponents, epsilon)
            if nargin > 0
                self.numComponents = numComponents;
            end
            if nargin > 1
                self.epsilon = epsilon;
            end
        end
    end

end

