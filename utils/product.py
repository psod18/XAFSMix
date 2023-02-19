"""Product iterator"""
from typing import List


class BadInitialValue(Exception):
    """Raise the error if initial value for Digit Wheel is greater than maximum """


class BadMaxWeightValue(Exception):
    """Raise the error if initial value for Digit Wheel is greater than maximum """


class DigitWheel:
    """Simulate digit wheel like in mechanism of  Combination Padlock"""
    
    def __init__(self, max_value: float, init_value: float = 0.0, step: float = 0.01):
        if init_value > max_value:
            raise BadInitialValue
        self.value = init_value
        self.step = step
        
        self._max_value = max_value
        self._init_value = init_value
        
        self.previous = None
        self.following = None
    
    def spin(self):
        if (self.value - self._max_value) > self.step/10:
            # spin previous digit wheel, if exist
            try:
                self.previous.spin()
            except AttributeError:
                raise StopIteration
            self.reset()
        else:
            self.value += self.step
            
    def reset(self):
        self.value = self._init_value

    def add_following(self, obj: "DigitWheel"):
        self.following = obj
        obj.previous = self


def coefficient_combiner(models_num: int, max_weight: float = 1.0, weight_step: float = 0.01):

    if models_num < 1:
        raise BadInitialValue("Number of models must be > 0")

    if 0.0 > max_weight > 1.0:
        raise BadMaxWeightValue("Weight coefficient threshold must belong [0,1)")

    padlock: List[DigitWheel] = []

    for wheel_idx in range(models_num):
        wheel = DigitWheel(max_weight, step=weight_step)
        if padlock:
            padlock[-1].add_following(wheel)
        padlock.append(wheel)

    while True:
        current_weight = [round(p.value, 2) for p in padlock]
        if round(sum(current_weight), 2) == round(max_weight, 2):
            yield current_weight
        try:
            padlock[-1].spin()
        except StopIteration:
            break


if __name__ == "__main__":

    c = coefficient_combiner(3, .33, weight_step=0.01)
    for j in c:
        print(j, sum(j))
