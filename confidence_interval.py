from scipy.stats import beta

def calc_confidence_interval(x, n, confidence_level=0.95):
    alpha_0, beta_0 = 1, 1

    alpha_post = alpha_0 + x
    beta_post = beta_0 + n - x

    lower_bound = beta.ppf((1 - confidence_level) / 2, alpha_post, beta_post)
    upper_bound = beta.ppf(1 - (1 - confidence_level) / 2, alpha_post, beta_post)

    return lower_bound, upper_bound


def calc_bool_confidence_interval(ratio, n, confidence_level=0.95):
    x = int(round(ratio * n))
    print(f"Success: {x}, Total: {n}")
    lower_bound, upper_bound = calc_confidence_interval(x, n, confidence_level)
    print(f"Success rate's {confidence_level*100}% confidence interval: ({lower_bound * 100:.0f}%,{upper_bound * 100:.0f}%)")


calc_bool_confidence_interval(0.067, 30)
