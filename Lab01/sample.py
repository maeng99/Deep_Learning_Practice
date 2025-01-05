import numpy as np

def print_array_stats(array):
  """Prints the mean, max, and min values of a numpy array.

  Args:
    array: The numpy array to print statistics for.
  """

  mean = np.mean(array)
  max_value = np.max(array)
  min_value = np.min(array)

  return mean, max_value, min_value