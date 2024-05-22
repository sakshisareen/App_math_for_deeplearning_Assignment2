import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    """
    return torch.full(dimensions, val)

def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    """
    return A * B

def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W, with W transposed for dimension matching.
    """
    return torch.matmul(X, W.T)

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W, add the bias, with W transposed.
    """
    product = torch.matmul(X, W.T)
    return product + b

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    """
    return torch.heaviside(sum_total, torch.tensor([0.0]))

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    """
    product_with_bias = calculate_matrix_prod_with_bias(X, W, b)
    return calculate_activation(product_with_bias)