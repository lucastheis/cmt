function setup_demo(varargin)
      %% Set up paths
      % The base cmt cpp implementition
      cmt_base = fileparts(mfilename('fullpath'));
      cmt_code = fullfile(cmt_base, 'code', 'cmt');
      cmt_src = fullfile(cmt_code, 'src');
      cmt_include = fullfile(cmt_code, 'include');

      % mex interface and wrapper
      mex_base = fullfile(cmt_code, 'matlab');
      mex_src = fullfile(mex_base, 'src');
      mex_include = fullfile(mex_base, 'include');

      % eigen matrix library
      eigen_include = fullfile(cmt_base, 'code');

      % liblbfgs L-BFGS library
      lbfgs_base = fullfile(cmt_base, 'code', 'liblbfgs');
      lbfgs_include = fullfile(lbfgs_base, 'include');
      lbfgs_lib = fullfile(lbfgs_base, 'lib', '.libs');
      lbfgs_obj = fullfile(lbfgs_lib, ['liblbfgs-1.10', system_dependent('GetSharedLibExt')]);

      % Build folder in temp
      [temp_base, rand_str] = fileparts(tempname);
      temp_out = fullfile(temp_base, ['cmt_build_', rand_str]);

      %% Check dependencies: check if liblbfgs was build
      assert(logical(exist(lbfgs_obj, 'file')), 'liblbfgs could not be found in ''%s''.', lbfgs_obj);

      %% Set up folder structur and move to target dir
      % Build folder
      mkdir(temp_out);
      temp_del = onCleanup(@() rmdir(temp_out,'s'));

      %% Define string helper functions
      toObjectName = @(s)regexprep(s, '\.cpp', '\.o');
      toMexName = @(s)regexprep(s, '\.cpp', ['\.', mexext()]);

      %% Source file lists for objects (normally used by multiple mex file)
      cmt_files = {'affinepreconditioner.cpp', ...
                   'affinetransform.cpp', ...
                   'binningtransform.cpp', ...
                   'conditionaldistribution.cpp', ...
                   'distribution.cpp', ...
                   'gsm.cpp', ...
                   'glm.cpp', ...
                   'mcgsm.cpp', ...
                   'mcbm.cpp', ...
                   'mixture.cpp', ...
                   'mlr.cpp', ...
                   'nonlinearities.cpp', ...
                   'patchmodel.cpp', ...
                   'pcapreconditioner.cpp', ...
                   'pcatransform.cpp', ...
                   'preconditioner.cpp', ...
                   'regularizer.cpp', ...
                   'stm.cpp', ...
                   'tools.cpp', ...
                   'trainable.cpp', ...
                   'utils.cpp', ...
                   'univariatedistributions.cpp', ...
                   'whiteningpreconditioner.cpp', ...
                   'whiteningtransform.cpp'};
               
      % Add absolute path to files
      cmt_files = fullfile(cmt_src, cmt_files);

      %% Specify default options
      default_options = {['-I', cmt_include], ...
                         ['-I', mex_include], ...
                         ['-I', eigen_include], ...
                         ['-I', lbfgs_include], ...
                         ['-L', lbfgs_lib], ...
                         '-llbfgs', ...                         '-largeArrayDims', ... 
                         'CXXFLAGS=$CXXFLAGS -std=c++0x'};
      if isunix() 
          if ~ismac()
            % Relative path linking
            default_options = [default_options, {'LDFLAGS=$LDFLAGS -Wl,-rpath,''$ORIGIN'' -Wl,-z,origin'}];
          else
            % Link against libc++ instead on libstdc++
            default_options = [default_options, {'CXXFLAGS=$CXXFLAGS -stdlib=libc++', '-lc++'}];              
          end
      end
      
      default_options = [default_options, varargin];  
      
      %% Build mex files
      demo_files = {'cmt_demo.cpp', 'cmt_demo_helper.cpp'};
      demo_files = fullfile(mex_src, demo_files);
      
      fprintf('Build in process ...'); tic();
      mex(default_options{:}, demo_files{:},  cmt_files{:});
      fprintf('.., done (%.2f).\n', toc());

      %% Copy all other files
      % Copy lbfgs library
      copyfile(lbfgs_obj, '.')

      % Change lbfgs library path on OS X
      if ismac()
          system(['install_name_tool -change "/usr/local/lib/liblbfgs-1.10.dylib" ', ...
                                            '"@loader_path/liblbfgs-1.10.dylib" ', ...
                                            '"', toMexName('cmt_demo.cpp'), '"']);
      end

      %% Profit!
      fprintf('\nSucessfully build demo!\n');
end
