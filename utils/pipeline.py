class Pipeline(object):
    """
    A Pipeline is an object representing the data transformations to make
    on the input data, finally outputting extracted features.

    pipeline: List of transforms to apply one by one to the input data
    """
    def __init__(self, pipeline):
        self.transforms = pipeline
        names = [t.get_name() for t in self.transforms]
        self.name = 'empty' if len(names) == 0 else '_'.join(names)

    def get_name(self):
        return self.name

    def apply(self, data):
        for transform in self.transforms:
            data = transform.apply(data)
        return data
