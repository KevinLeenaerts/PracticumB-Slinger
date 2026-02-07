"""
CURVE_FIT functie voor practicum natuurkunde

Auteur: 	Bob Stienen
Datum:		9 februari 2024

-------------------------------------------------------------------------------
"""

from scipy import odr
import numpy as np
from inspect import signature


class CurveFitter:
	def __init__(self, curve):
		self.curve = curve
		self.input = {}
		self.target = np.array([])
		self.target_error = np.array([])
		self.estimates = {}
		self.maxit = 1000000
		self.results = None
	
	def add_measurements(self, x, dx, name):
		x = self._convert_to_numpy(x)
		dx = self._convert_to_numpy(dx)
		self._check_internal_dimensionality(x, dx)
		self.input[name] = [x, dx]
	
	def set_target_data(self, y, dy):
		y = self._convert_to_numpy(y)
		dy = self._convert_to_numpy(dy)
		self._check_internal_dimensionality(y, dy)
		self.target, self.target_error = y, dy
	
	def add_free_parameter(self, name, estimate):
		self.estimates[name] = estimate
	
	def limit_numer_of_iterations(self, number=10.000):
		self.maxit = number
	
	def remove_iteration_limit(self):
		self.maxit = None
	
	def fit(self):
		self._check_argument_count()
		self._check_data_consistency()
		model = self._construct_model()
		data = self._construct_data()
		estimates = self._construct_estimates()
		analysis = odr.ODR(data, model, beta0=estimates, maxit=self.maxit)
		result = analysis.run()

		parameter_names = list(self.estimates.keys())
		return FitResult(result, parameter_names)

	def _construct_model(self):
		def func(args, x):
			if len(x.shape) == 1:
				x = x.reshape(1, -1)
			if x.shape[0] != len(self.input):
				vars = [x[i] for i in range(len(x))]
			else:
				vars = x
			return self.curve(*vars, *args)
		return odr.Model(func)

	def _construct_data(self):
		x, dx = [], []
		for name in self.input:
			x.append(self.input[name][0])
			dx.append(self.input[name][1])
		x = np.hstack(x).T
		dx = np.hstack(dx).T
		return odr.RealData(x, self.target.T, dx, self.target_error.T)

	def _construct_estimates(self):
		return [self.estimates[name] for name in self.estimates]
	
	def _check_curve_function(self):
		if not callable(self.curve):
			raise ValueError("The curve you want to fit is not callable. Make sure you supply it as a function handle (i.e. a function without parentheses).")

	def _convert_to_numpy(self, iterable):
		return np.array(iterable).reshape(-1,1)
	
	def _check_internal_dimensionality(self, values, deltas):
		if values.shape != deltas.shape:
			raise ValueError("The shape of your inputs do not match. Make sure your data and uncertainties have the same length and size.")
	
	def _check_data_consistency(self):
		k = list(self.input.keys())[0]
		if len(self.target) != len(self.input[k][0]):
			raise ValueError("You should supply as many target values as you supplied data points for each of the input parameters")

	def _check_argument_count(self):
		if len(self.input) + len(self.estimates) != len(signature(self.curve).parameters):
			raise ValueError("The number of supplied variables (input data + estimates) is not equal to the number of arguments of the function you want to fit.")


class FitResult():
	def __init__(self, odr_output, variable_names):
		self.odr_output = odr_output
		self.names = variable_names
	
	def get_parameters(self):
		return {self.names[i]: self.odr_output.beta[i] for i in range(len(self.names))}

	def get_uncertainties(self):	
		covmat = self.get_covariance_matrix()
		sigmas = np.sqrt(np.diag(covmat))
		return {self.names[i]: sigmas[i] for i in range(len(self.names))}

	def get_result(self, name):
		for i,n in enumerate(self.names):
			if n == name:
				covmat = self.get_covariance_matrix()
				sigmas = np.sqrt(np.diag(covmat))
				return {'fit': self.odr_output.beta[i], 'uncertainty': sigmas[i]}

	def get_stop_reason(self):
		return self.odr_output.stopreason

	def get_covariance_matrix(self):
		return self.odr_output.cov_beta


