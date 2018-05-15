class BinaryManager(object):
 	"""docstring for BinaryManager"""
 	def __init__(self):
 		super(BinaryManager, self).__init__()
 		self.dictionary = {}
 		self._diff_element = 0

 	def _add_items(self, item):
 		string = ''
 		for itr in item:
 			if itr < 0:
 				string += '0'
 			else:
 				string += '1'
 		if string in self.dictionary:
 			self.dictionary[string] += 1
 		else:
 			self.dictionary[string] = 1
 			self._diff_element += 1
 		print string, self.dictionary[string]
 		print 'Different: ', self._diff_element


