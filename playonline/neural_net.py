import math

class FlappyNeuralNet:
    """
    Feed-forward Neural Network with 0 hidden layers (Perceptron)
    using the 3 exact inputs:
        inputs = (bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom))
    """
    def __init__(self,
                 bias=-1.275138,
                 w_bird_y=-0.040200,
                 w_dist_top=0.401631,
                 w_dist_bottom=-0.700470,
                 jump_threshold=0.25):
        self.bias = bias
        self.w_bird_y = w_bird_y
        self.w_dist_top = w_dist_top
        self.w_dist_bottom = w_dist_bottom
        self.jump_threshold = jump_threshold

    def activate(self, inputs):
        bird_y, dist_top, dist_bottom = inputs
        z = (self.bias
             + self.w_bird_y * bird_y
             + self.w_dist_top * dist_top
             + self.w_dist_bottom * dist_bottom)
        return [math.tanh(z)]

    def should_jump(self, inputs):
        output = self.activate(inputs)
        return output[0] > self.jump_threshold
