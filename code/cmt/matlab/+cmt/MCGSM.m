classdef MCGSM < cmt.Trainable
    %MCGSM is an implementation of a mixture of conditional Gaussian scale mixtures.
    %   The distribution defined by the model is
    %       $p(y \mid x) = \sum_{c, s} p(c, s \mid x) p(y \mid c, s, x),$
    %
    %   To create an MCGSM for $N$-dimensional inputs $x \in R^N$ and $M$-dimensional outputs $y \in R^M$ with, for example, 8 predictors $A_c$, 6 scales $\alpha_{cs}$ per component $c$, and 100 features $b_i$, use
    %       model = cmt.MCGSM(N, M, 8, 6, 100)
    %
    %    <a href="matlab: doc cmt.MCGSM">Function and Method Overview</a>
    %
    %   See also: cmt.Trainable, cmt.ConditionalDistribution

    properties (SetAccess = private)
        % Number of predictors.
        numComponents;
        % Number of features available to approximate input covariances.
        numFeatures;
        % Number of scale variables per component.
        numScales;
    end

    properties
        % A list of Cholesky factors of residual precision matrices, $L_c$.
        choleskyFactors;
        % Features used for capturing structure in inputs, $b_i$.
        features;
        % Linear features, $w_c$.
        linearFeatures
        % Means of outputs, $u_c$.
        means;
        % A list of linear predictors, $A_c$.
        predictors;
        % Log-weights of mixture components and scales, $\eta_{cs}$.
        priors;
        % Log-precision variables, $\alpha_{cs}$.
        scales;
        % Weights relating features and mixture components, $\beta_{ci}$.
        weights;
    end

    methods
        function self = MCGSM(dimIn, varargin)
            %MCGSM creates a new MCGSM object.
            %   Parameters:
            %       dimIn - dimensionality of input
            %       dimOut (optional) - dimensionality of output (default: 1)
            %       numComponents (optional) - number of components (default: 8)
            %       numScales (optional) - number of scales per scale mixture component (default: 6)
            %       numFeatures (optional) - number of features used to approximate input covariance matrices (default: dimIn)
            %   Returns:
            %       a new MCGSM object
            self@cmt.Trainable(dimIn, varargin{:});
        end


        % Constant properties
        function v = get.numComponents(self)
            v = self.mexEval('numComponents');
        end

        function v = get.numFeatures(self)
            v = self.mexEval('numFeatures');
        end

        function v = get.numScales(self)
            v = self.mexEval('numScales');
        end


        % Nonconstant properties
        function set.choleskyFactors(self, v)
            self.mexEval('setCholeskyFactors', v);
        end

        function v = get.choleskyFactors(self)
            v = self.mexEval('choleskyFactors');
        end


        function set.features(self, v)
            self.mexEval('setFeatures', v);
        end

        function v = get.features(self)
            v = self.mexEval('features');
        end


        function set.linearFeatures(self, v)
            self.mexEval('setLinearFeatures', v);
        end

        function v = get.linearFeatures(self)
            v = self.mexEval('linearFeatures');
        end


        function set.means(self, v)
            self.mexEval('setMeans', v);
        end

        function v = get.means(self)
            v = self.mexEval('means');
        end


        function set.predictors(self, v)
            self.mexEval('setPredictors', v);
        end

        function v = get.predictors(self)
            v = self.mexEval('predictors');
        end


        function set.priors(self, v)
            self.mexEval('setPriors', v);
        end

        function v = get.priors(self)
            v = self.mexEval('priors');
        end


        function set.scales(self, v)
            self.mexEval('setScales', v);
        end

        function v = get.scales(self)
            v = self.mexEval('scales');
        end


        function set.weights(self, v)
            self.mexEval('setWeights', v);
        end

        function v = get.weights(self)
            v = self.mexEval('weights');
        end


        % Methods
        function value = posterior(self, input, output)
            %POSTERIOR computes the posterior distribution over component labels, $p(c \mid x, y)$
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns
            %   Returns:
            %       a posterior distribution over labels for each given pair of input and output
            value = self.mexEval('posterior', input, output);
        end

        function value = prior(self, input)
            %PRIOR computes the prior distribution over component labels, $p(c \mid x)$
            %   Parameters:
            %       input - inputs stored in columns
            %   Returns:
            %       a distribution over labels for each given input
            value = self.mexEval('prior', input);
        end

        function value = sample(self, input, varargin)
            % SAMPLE generates outputs for given inputs. If labels are specified, uses the given mixture component to generate outputs.
            %   Parameters:
            %       input - inputs stored in columns
            %       labels (optional) - indices indicating mixture components
            %   Returns:
            %       sampled outputs
            value = self.mexEval('sample', input, varargin{:});
        end

        function value = samplePosterior(self, input, output)
            %SAMPLEPOSTERIOR samples component labels $c$ from the posterior $p(c \mid x, y)$.
            %   Parameters:
            %       input - inputs stored in columns
            %       output - outputs stored in columns
            %   Returns:
            %       an integer array containing a sampled index for each input and output pair
            value = self.mexEval('samplePosterior', input, output);
        end

        function value = samplePrior(self, input)
            %SAMPLEPRIOR samples component labels $c$ from the distribution $p(c \mid x)$.
            %   Parameters:
            %       input - inputs stored in columns
            %   Returns:
            %       an integer array containing a sampled index for each input and output pair
            value = self.mexEval('samplePrior', input);
        end
    end

    properties (Constant, Hidden)
        constructor_arguments = {'dimIn', 'dimOut', 'numComponents', ...
                                 'numScales', 'numFeatures'};
    end

    methods (Static)
        function obj = loadobj(S)
            obj = cmt.MCGSM.mexLoad(S, @cmt.MCGSM, cmt.MCGSM.constructor_arguments);
        end
    end
end
