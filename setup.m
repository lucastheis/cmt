function setup(varargin)
      %% Current version
      curr_ver = '1.0.0';

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
      if ispc()
        lbfgs_lib = fullfile(lbfgs_base, 'x64', 'Release');
        lbfgs_obj = fullfile(lbfgs_lib, 'lbfgs.lib');
      else
        lbfgs_lib = fullfile(lbfgs_base, 'lib', '.libs');
        lbfgs_obj = fullfile(lbfgs_lib, ['liblbfgs-1.10', system_dependent('GetSharedLibExt')]);
      end
        
      % Build folder in temp
      [temp_base, rand_str] = fileparts(tempname);
      temp_out = fullfile(temp_base, ['cmt_build_', rand_str]);

      % target directory
      distrib_out = fullfile(cmt_base, 'distribute');
      package_out = fullfile(distrib_out, '+cmt');

      %% Check dependencies: check if liblbfgs was build
      assert(logical(exist(lbfgs_obj, 'file')), 'liblbfgs could not be found in ''%s''.', lbfgs_obj);

      %% Define string helper functions
      if ispc()
          toObjectName = @(s)regexprep(s, '\.cpp', '\.obj');
      else
          toObjectName = @(s)regexprep(s, '\.cpp', '\.o');
      end
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

      cmt_obj = toObjectName(cmt_files);

      % Add absolute path to files
      cmt_files = fullfile(cmt_src, cmt_files);

      cmt_obj = fullfile(temp_out, cmt_obj);

      mex_hpp_files = {'mexinput.cpp', ...
                       'mexoutput.cpp',...
                       'mexdata.cpp'};

      mex_hpp_obj = toObjectName(mex_hpp_files);

      % Add absolute path to files
      mex_hpp_files = fullfile(mex_src, mex_hpp_files);

      mex_hpp_obj = fullfile(temp_out, mex_hpp_obj);

      train_files = {'conditionaldistributioninterface.cpp', ...
                     'trainableinterface.cpp', ...
                     'nonlinearitiesinterface.cpp',...
                     'univariatedistributionsinterface.cpp',...
                     'regularizerinterface.cpp'};

      train_obj = toObjectName(train_files);

      % Add absolute path to files
      train_files = fullfile(mex_src, train_files);

      train_obj = fullfile(temp_out, train_obj);

      %% List of mex interface files (each interface = one mex file)
      trainable_intefaces = {'glminterface.cpp', ...
                             'stminterface.cpp', ...
                             'mcgsminterface.cpp'};


      % Check if any of the already exist and are lock (i.e. there are unfreed object of that interface)
      for mex_file = toMexName(trainable_intefaces)
          if mislocked(fullfile(distrib_out, '+cmt', mex_file{:}))
                  error('The mexfile "%s" seems to be locked. Please delete all associated cmt object and try again.', fullfile(distrib_out, '+cmt', mex_file{1}));
          end
      end

      %% Specify default options
      default_options = {['-I', cmt_include], ...
                         ['-I', mex_include], ...
                         ['-I', eigen_include], ...
                         ['-I', lbfgs_include], ...
                         ['-L', lbfgs_lib], ...
                         '-llbfgs', ...
                         '-largeArrayDims', ...
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

      %default_options = [default_options, varargin];

      %% Disable warning on Linux
      warning('off', 'MATLAB:mex:GccVersion_link');

      %% Create shared object files
      % Build folder
      mkdir(temp_out);
      temp_del = onCleanup(@() rmdir(temp_out,'s'));
      
      % Run mex compiler
      mex('-outdir', temp_out, '-c', default_options{:}, cmt_files{:});
      mex('-outdir', temp_out, '-c', default_options{:}, train_files{:});
      mex('-outdir', temp_out, '-c', default_options{:}, mex_hpp_files{:});

      %% Build mex files
      % Create folders
      if ~exist(distrib_out, 'dir')
        mkdir(distrib_out);
      end
      if ~exist(package_out, 'dir')
        mkdir(package_out)
      end
      
      % Run mex compiler
      for mex_file = trainable_intefaces
            mex_file =  fullfile(mex_src, mex_file{1}); %#ok<FXSET>
            mex('-outdir', package_out, default_options{:}, mex_file, train_obj{:}, mex_hpp_obj{:}, cmt_obj{:});
      end

      %% Copy all other files
      % Copy m files
      copyfile(fullfile(mex_base, '+cmt', '*.m'), package_out);

      % Copy example and test script files
      copyfile(fullfile(mex_base,'test.m'), distrib_out);
      copyfile(fullfile(mex_base,'callback_test.m'), distrib_out);

      % Copy lbfgs library
      copyfile(lbfgs_obj, package_out)

      % Change lbfgs library path on OS X
      if ismac()
            for mex_file = toMexName(trainable_intefaces)
                  mex_file = fullfile(package_out, mex_file{1}); %#ok<FXSET>
                  system(['install_name_tool -change "/usr/local/lib/liblbfgs-1.10.dylib" ', ...
                                                    '"@loader_path/liblbfgs-1.10.dylib" ', ...
                                                    '"', mex_file ,'"']);
            end
      end

      %% Zip resulting file
      file_list = {package_out, ...
                   fullfile(distrib_out, 'test.m'), ...
                   fullfile(distrib_out, 'callback_test.m')};
      zip_file = ['cmt-matlab_', curr_ver, '_', computer('arch'), '.zip'];
      zip(zip_file, file_list, cmt_base);
      
      %% Profit!
      fprintf(['\nSucessfully built mex extension. ', ...
               'Copy the content of "%s" to your project folder ', ...
               'or add it to your Matlab path to be able ', ...
               'to use the "Conditional Modeling Toolkit" in matlab.\n'], distrib_out);
end
