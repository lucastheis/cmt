classdef Trainable < cmt.ConditionalDistribution
    methods
        function self = Trainable(varargin)
            self@cmt.ConditionalDistribution(varargin{:});
        end

        function initialize(self, input, output)
            %INITIALIZE tries to guess more sensible initial values for the model parameters from data.
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns

            self.mexEval('initialize', input, output);
        end

        function bool = train(self, input, output, varargin)
            %TRAIN fits model parameters to given data using L-BFGS.
            %   The following example demonstrates possible parameters and default settings.
            %       model.train(input, output, ...
            %           'verbosity', 0, ...
            %           'max_iter', 1000, ...
            %           'threshold', 1e-9, ...
            %           'num_grad', 20, ...
            %           'batch_size', 2000, ...
            %           'callback', None, ...
            %           'cb_iter', 25, ...
            %           'val_iter', 5, ...
            %           'val_look_ahead', 20, ...
            %           'train_biases', true, ...
            %           'train_weights', true, ...
            %           'train_features', true, ...
            %           'train_predictors', true, ...
            %           'train_linear_predictor', true)
            %
            %   The parameters train_biases, train_weights, and so on can be
            %   used to control which parameters will be optimized.
            %   Optimization stops after max_iter iterations or if the
            %   difference in (penalized) log-likelihood is sufficiently
            %   small enough, as specified by threshold. num_grad is the
            %   number of gradients used by L-BFGS to approximate the inverse
            %   Hessian matrix.
            %   Regularization of parameters $z$ adds a penalty term
            %   $\eta ||A z||_p$ to the average log-likelihood, where $\eta$
            %   is given by strength, $A$ is given by transform, and $p$ is
            %   controlled by norm, which has to be either 'L1' or 'L2'. The
            %   parameter batch_size has no effect on the solution of the
            %   optimization but can affect speed by reducing the number of
            %   cache misses.
            %   If a callback function is given, it will be called every
            %   cb_iter iterations. The first argument to callback will be
            %   the current iteration, the second argument will be a copy of
            %   the model.
            %
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns
            %       input_val (optional) - inputs used for early stopping based on validation error
            %       output_val (optional) - outputs used for early stopping based on validation error
            %       parameters (optional)  - additional hyperparameters (see above)
            %   Returns:
            %       True if training converged, otherwise False
            bool = self.mexEval('train', input, output, varargin{:});
        end

        function value = checkGradient(self, input, output, varargin)
            %CHECKGRADIENT compare the gradient to a numerical gradient.
            %   Numerically estimate the gradient using finite differences
            %   and return the norm of the difference between the numerical
            %   gradient and the gradient used during training. This method
            %   is used for testing purposes.
            %   Parameters:
            %       input - inputs stored in columns
            %       output - inputs stored in columns
            %       epsilon (optional) - a small change added to the current parameters
            %       parameters (optional) - additional hyperparameters
            %   Returns:
            %       difference between numerical and analytical gradient
            value = self.mexEval('checkGradient', input, output, varargin{:});
        end

        function value = checkPerformance(self, input, output, varargin)
            %CHECKPERFORMANCE measures the time it takes to evaluate the parameter gradient for the given data points.
            %   This function can be used to tune the batch_size parameter.
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns
            %       repetitions (optional) - number of times the gradient is evaluated before averaging
            %       parameters (optional) - additional hyperparameters
            %   Returns:
            %       estimated time for one gradient evaluation
            value = self.mexEval('checkPerformance', input, output, varargin{:});
        end

        function matrix = fisherInformation(self, input, output)
            % source code
            %   Estimates the Fisher information matrix of the parameters.
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns
            %       parameters (optional) - additional hyperparameters
            %   Returns:
            %       the Fisher information matrix
            matrix = self.mexEval('fisherInformation', input, output);
        end
    end
end
