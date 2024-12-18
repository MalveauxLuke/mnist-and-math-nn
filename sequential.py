# this will be the main structure of our network
class Sequential:
    def __init__(self):
        # storage for all the layers in our net. Accessible through add and softmax functions
        self.layers = []
        self.softmax_layer = []

    def add(self, layer):
        # appends to end of layers list above
        self.layers.append(layer)

    def softmax(self, layer):
        # appends to end of softmax_layer list above
        self.softmax_layer.append(layer)

    def forward(self, inputs):
        # set output to input
        output = inputs
        # for every layer in our vector we go through this loop
        for layer in self.layers:
            # we input the input starting with our raw data. Everytime we do a forward iteration set output = to the ouput and pass it into the next layer
            output = layer.forward(output)
        return output

    # used to get final prediction Will adjust to become more modular in the future. Right now in this version it
    # only needs to be used for softmax output.
    def prediction(self, test):
        output = test
        output = self.forward(output)
        if len(self.softmax_layer) > 0:
            prediction = self.softmax_layer[0].forward(output)
        else:
            pass
        return prediction

    # starts with the output gradient. exact same as forward but reversed
    def backwards(self, output_gradient, learning_rate):
        inputs = output_gradient
        for layer in reversed(self.layers):
            inputs = layer.backwards(inputs, learning_rate)
