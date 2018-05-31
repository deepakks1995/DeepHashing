class BinaryManager(object):
    """docstring for BinaryManager"""
    def __init__(self):
        super(BinaryManager, self).__init__()
        self.dictionary = {}
        self._diff_element = 0

    def process_batch(self, model_vars, idx_list):
        self.dictionary = {}
        self._diff_element = 0
        set1 = set()
        for itr in range(len(idx_list)):
            set1.add(idx_list[itr][0])

        for index in set1:
            self._add_items(model_vars.B[:, index])

        print ('Different/Total ', self._diff_element, "/", len(set1))

    def process_dataset(self, model_vars):
        self.dictionary = {}
        self._diff_element = 0
        for itr in range(model_vars._length):
            index = itr*model_vars._samples
            self._add_items(model_vars.B[:, index])
        print ()
        print ('On Epoch Complete Different/Total ', self._diff_element, "/", model_vars.total_images)
        return self._diff_element

    def _add_items(self, item):
        string = ''
        for itr in item:
            if itr <= 0:
                string += '0'
            else:
                string += '1'
        if string in self.dictionary:
            self.dictionary[string] += 1
        else:
            self.dictionary[string] = 1
            self._diff_element += 1
        print ("String ", string)