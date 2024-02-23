def integrate_summation_rule(f, a, b, num_points):
    """Compute the integral of a function using the summation rule.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.
    num_points : int
        The number of sample points to use.
    Returns
    -------
    float
        The value of the integral of the function over the interval [a, b].
    """

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


def integrate_double_summation_rule(f, a, b, c, d, num_points):
    """Compute the integral of a function using the summation rule.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.
    num_points : int
        The number of sample points to use.
    Returns
    -------
    float
        The value of the integral of the function over the interval [a, b].
    """

    # Calculate the width of each interval
    dx = (b - a) / num_points
    dy = (d - c) / num_points

    # Initialize integral value
    integral = 0

    # Iterate over all sample points within the interval [a, b]
    for i in range(num_points):
        # Calculate the x-value for the current sample point
        x = a + i * dx
        for j in range(num_points):
            y = c + j * dy
            # Evaluate the function at the current sample point
            z = f(x, y)

            # Add the area of the rectangle formed by the function value and the width of the interval
            integral += z * dx * dy

    return integral