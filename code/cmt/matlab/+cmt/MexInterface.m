% C++ Object Wrapper
classdef (Abstract) MexInterface < matlab.mixin.Copyable
    properties (Access = private, Hidden = true, Transient = true)
        mexHandle;
        mexInterface;
    end

    methods
        function self = MexInterface(varargin)
            self.mexInterface = eval(['@', lower(class(self)), 'interface']);
            if isa(varargin{1}, 'uint64') && length(varargin) == 1
                if feval(self.mexInterface, 'validate', varargin{1})
                    self.mexHandle = varargin{1};
                else
                    error('MexInterface:uint64NotValidHandle', 'UInt64 received is not a valid handle.')
                end
            else
                self.mexHandle = self.mexInterface('new', varargin{:});
            end
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
            if strcmp(cmd, {'new', 'delete', 'validate', 'copy'})
                error('MexInterface:protectedCommand', '"%s" is a protected command and you are not allowed to call this directly.', cmd)
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

        function obj = copyElement(self)
            obj = copyElement@matlab.mixin.Copyable(self);
            obj.mexHandle = obj.mexInterface('copy', self.mexHandle);
        end
    end
    methods (Static, Access = protected)
        function obj = mexLoad(S, constructor, arg_names, dep_name)
            %MEXLOAD helps turning Matlab save file back to C++ object.
            % It does that by running the function "constructor" using
            % fields from struct "S" named by "arg_names". Afterwards it
            % checks if all dependent properties were set correctly and
            % load all the other properties left in S.
            args = cell(1, length(arg_names));
            for i = 1:length(arg_names)
                args{i} = S.(arg_names{i});
                S = rmfield(S, arg_names{i});
            end
            obj = constructor(args{:});
            if nargin > 3
                for i = 1:length(dep_name)
                    if obj.(dep_name{i}) ~= S.(dep_name{i})
                        warning('Could not load dependent property ''%s'' correctly.', dep_name{i})
                    end
                    S = rmfield(S, dep_name{i});
                end
            end
            for field = fieldnames(S)'
                obj.(field{1}) = S.(field{1});
            end

        end
    end
end
