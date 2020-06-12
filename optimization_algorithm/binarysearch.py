import math


def fsolve(func, lower, upper, tol=1e-7, max_iteration=1000):
    if func(lower) == 0:
        return lower
    elif func(upper) == 0:
        return upper

    if (func(upper) < 0) == (func(lower) < 0):
        raise ValueError("Function does not have a valid solution or is not monotonic function.")
    elif func(upper) <= 0 and func(lower) >= 0:
        lower, upper = upper, lower

    # Binary Search
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        mid = (lower + upper) / 2
        if abs(func(mid)) <= tol: 
            print("Iteration: %d" % iteration)
            return mid

        elif func(mid) > 0:
            upper = mid
        else:
            lower = mid

    print("Maximum iteration %d reached." % max_iteration)
    return


# Test.
def func(x):
    return 3*math.exp(0.5*x) - 0.5*math.exp(3*x) - x - 1


sol = fsolve(func, -10, 10)

print("Solution is %.5f" % sol)