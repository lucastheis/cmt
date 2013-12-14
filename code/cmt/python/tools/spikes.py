"""
Tools for dealing with spike trains.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '0.1.0'

from cmt.tools import extract_windows

def generate_data_from_spike_train(stimulus, stimulus_history, spike_train=None, spike_history=0):
	"""
	Extracts windows from a stimulus time-series and a spike train.

	@type  stimulus: C{ndarray}
	@param stimulus: time-series (NxT) where the second dimension is time

	@type  stimulus_history: C{int}
	@param stimulus_history: length of extracted stimulus windows

	@type  spike_train: C{ndarray}
	@param spike_train: spikes corresponding to stimulus (1xT)

	@type  spike_history: C{int}
	@param spike_history: length of extracted spike windows

	@rtype: C{tuple}/C{ndarray}
	@return: stimulus windows, spike histories, and spikes
	"""

	if stimulus.ndim == 1:
		stimulus = stimulus.reshape(1, -1)

	if spike_train is None:
		return extract_windows(stimulus, stimulus_history)

	if spike_train.ndim == 1:
		spike_train = spike_train.reshape(1, -1)

	if stimulus.shape[1] != spike_train.shape[1]:
		raise ValueError('Stimulus and spike train should have the same length.')

	# extract stimulus and spike history windows
	spikes = extract_windows(spike_train, spike_history + 1)
	stimuli = extract_windows(stimulus, stimulus_history)

	# make sure stimuli and spikes are aligned
	spikes = spikes[:, -stimuli.shape[1]:]
	stimuli = stimuli[:, -spikes.shape[1]:]

	# separate last spike of each window
	outputs = spikes[[-1]]
	spikes = spikes[:-1]

	if spike_history > 0:
		return stimuli, spikes, outputs
	else:
		return stimuli, outputs
