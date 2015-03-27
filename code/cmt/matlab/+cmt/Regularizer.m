classdef Regularizer

    properties
        strength = 0;
        norm = 'L2';
    end

    methods
        function self = Regularizer(strength, norm)
            if nargin > 0
                self.strength = strength;
            end

            if nargin > 1
                assert(any(strcmp(norm, {'L1', 'L2'})), 'Unknown norm ''%s''. Must be either ''L1'' or ''L2''.', norm);
                self.norm = norm;
            end

        end
    end

end

