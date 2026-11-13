import math


def learning_rate_factor(step):
    """Absolute-step schedule, unchanged when the user later extends the run."""
    if step <= 7500:
        return 1.
    phase = min(1., (step-7500)/5000.)
    return .2 + .8 * .5 * (1 + math.cos(math.pi*phase))

