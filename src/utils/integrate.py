
def integrate_summation_rule(f, a, b, num_points):
    # Calculate the width of each interval
    dx = (b - a) / num_points

    # Initialize integral value
    integral = 0

    # Iterate over all sample points within the interval [a, b]
    for i in range(num_points):
        # Calculate the x-value for the current sample point
        x = a + i * dx

        # Evaluate the function at the current sample point
        y = f(x)

        # Add the area of the rectangle formed by the function value and the width of the interval
        integral += y * dx

    return integral
