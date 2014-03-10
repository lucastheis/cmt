function setup()
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

      % target directory
      distrib_out = fullfile(cmt_base, 'distribute');

      %% Check dependencies: check if liblbfgs was build
      assert(logical(exist(lbfgs_obj, 'file')), 'liblbfgs could not be found in ''%s''.', lbfgs_obj);

      %% Set up folder structur and move to target dir
      % Build folder
      mkdir(temp_out);
      temp_del = onCleanup(@() rmdir(temp_out,'s'));

      % Target dir
      mkdir(distrib_out);
      old_pwd = cd(distrib_out);
      pwd_reset = onCleanup(@() cd(old_pwd));

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


      train_files = { 'conditionaldistributioninterface.cpp', ...
                      'trainableinterface.cpp'};

      train_obj = toObjectName(train_files);

      % Add absolute path to files
      train_files = fullfile(mex_src, train_files);

      train_obj = fullfile(temp_out, train_obj);

      %% List of mex interface files (each interface = one mex file)
      trainable_intefaces = {'glminterface.cpp', ...
                             'stminterface.cpp'};


      % Check if any of the already exist and are lock (i.e. there are unfreed object of that interface)
      for mex_file = toMexName(trainable_intefaces)
            if mislocked(fullfile(distrib_out, '+cmt', mex_file{:}))
                  error('The mexfile "%s" seems to be locked. Please deleta all associated cmt object and try again.', fullfile(distrib_out, '+cmt', mex_file{1}));
          end
      end

      %% Specify default options
      default_options = {['-I', cmt_include], ...
                         ['-I', mex_include], ...
                         ['-I', eigen_include], ...
                         ['-I', lbfgs_include], ...
                         ['-L', lbfgs_lib], ...
                         '-llbfgs', ...
                         '-largeArrayDims', 'CXXFLAGS=""\$CXXFLAGS -std=c++0x""', ...
                         '-v'};

      if isunix() && ~ismac()
            % Relative path linking
            default_options = [default_options, {'-Wl,-rpath,''\$ORIGIN''', '-Wl,-z,origin'}];
      end

      %% Create shared object files
      mex('-outdir', temp_out, '-c', default_options{:}, cmt_files{:});
      mex('-outdir', temp_out, '-c', default_options{:}, train_files{:});

      %% Build mex files
      mkdir('+cmt')
      for interface = trainable_intefaces
            mex('-outdir', '+cmt', default_options{:}, fullfile(mex_src, interface{1}), train_obj{:}, cmt_obj{:});
      end

      %% Copy all other files
      % Copy m files
      copyfile(fullfile(mex_base, '+cmt', '*.m'), '+cmt');

      % Copy example and test script files
      copyfile(fullfile(mex_base,'test.m'), '.');

      % Copy lbfgs library
      copyfile(lbfgs_obj, '+cmt')

      % Change lbfgs library path on OS X
      if ismac()
            for mex_file = toMexName(trainable_intefaces)
                  system(['install_name_tool -change "/usr/local/lib/liblbfgs-1.10.dylib" ', ...
                                                    '"@loader_path/liblbfgs-1.10.dylib" ', ...
                                                    '"+cmt/', mex_file{1}, '"']);
            end
      end

      %% Profit!
      fprintf(['\nSucessfully build mex extension. ', ...
               'Copy the content of "%s" to your project folder ', ...
               'or add it to your Matlab path to be able ', ...
               'to use the "Conditional Modeling Toolkit" in matlab.\n'], distrib_out);
end
