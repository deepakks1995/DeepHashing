import itertools

class BinaryManager(object):
    """docstring for BinaryManager"""
    def __init__(self):
        super(BinaryManager, self).__init__()
        self.dictionary = {}
        self._diff_element = 0

    def process_batch(self, model_vars, batch):
        self.dictionary = {}
        self._diff_element = 0
        set1 = set()
        for itr in batch:
            for tuples in itertools.chain(itr[2], itr[3]):
                index = tuples[0]*model_vars._samples + tuples[1]
                set1.add(index)
        for index in set1:
            self._add_items(model_vars.B[:, index])
        print ('Different:Total ', self._diff_element),
        print ("/", len(set1) )

    def _add_items(self, item):
        string = ''
        for itr in item:
            if itr == 0:
                string += '0'
            else:
                string += '1'
        if string in self.dictionary:
            self.dictionary[string] += 1
        else:
            self.dictionary[string] = 1
            self._diff_element += 1
        # print "String ", string