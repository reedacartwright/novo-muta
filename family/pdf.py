#!/usr/bin/env python
import scipy.special as sp

def dirichlet_multinomial(alpha, n):
    """
    Calculate probability from the pdf
        \frac{\gamma(\theta)}{\gamma(\theta + N)} *
            \Pi_{i = A, C, G, T} \frac{\gamma(\alpha_i * \theta + n_i)}
                                      {\gamma(\alpha_i * \theta}

    Input
    -----
    alpha - list of frequencies (doubles) for each category in the multinomial
            that sum to one 
    n     - list of samples (integers) for each category in the multinomial 

    Output
    ------
    proba - a double equal to log_e(P) where P is the value calculated
            from the pdf above

    Notes
    -----
    We refer to the first term in the product as the constant_term 
    (because its value doesn't vary with the number of nucleotide counts) and
    the second term in the product as the product_term
    """
    N = sum(n)
    A = sum(alpha)
    constant_term = (sp.gammaln(A) - sp.gammaln(N + A))
    product_term = 0
    for i in range(len(n)):
        product_term += (sp.gammaln(alpha[i] + n[i]) - sp.gammaln(alpha[i]))
    return constant_term + product_term  # log_proba