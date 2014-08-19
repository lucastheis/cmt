% C++ Object Wrapper
classdef (Abstract) MexInterface < handle
    properties (Access = private, Hidden = true)
        mexVarargin;
    end
    properties (Access = private, Hidden = true, Transient = true)
        mexHandle;
        mexInterface;
    end

    methods
        function self = MexInterface(varargin)
            self.mexInterface = eval(['@', lower(class(self)), 'interface']);
            self.mexHandle = self.mexInterface('new', varargin{:});
            self.mexVarargin = varargin;
        end

        function varargout = debug(self, cmd, varargin)
            [varargout{1:nargout}] = self.mexEval(cmd, varargin{:});
        end

        function delete(self)
            if ~isempty(self.mexHandle)
                self.mexInterface('delete', self.mexHandle);
                self.mexHandle = [];
            end
        end
    end

    methods (Access = protected, Sealed = true)
        % Set data to mex interface for evaluation
        function varargout = mexEval(self, cmd, varargin)
            if strcmp(cmd, {'new', 'delete'})
                error('"%s" is a protected command and you are not allowed to call this directly.', cmd)
            end

            % Unwrap all other MexInterface based objects before they are
            % parsed
            for i = 1:(nargin - 2)
                if isa(varargin{i}, 'MexInterface')
                    varargin{i} = varargin{i}.mexHandle;
                end
            end

            [varargout{1:nargout}] = self.mexInterface(cmd, self.mexHandle, varargin{:});
        end
    end
    methods (Static, Access = protected)
        function obj = mexLoad(S, constructor)
            obj = constructor(S.mexVarargin{:});
            S = rmfield(S, 'mexVarargin');
            for field = fieldnames(S)'
                obj.(field{1}) = S.(field{1});
            end
        end
    end
end
