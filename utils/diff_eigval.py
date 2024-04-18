import torch
from memory_profiler import profile
import psutil


class LargestEigenvalueFunction(torch.autograd.Function):
    @staticmethod
    # @profile
    def forward(ctx, grad_func, x, num_iterations=10, tol=1e-4) -> torch.Tensor:
        # Compute the largest eigenvalue using power iteration method
        largest_eigenvalue = torch.ones([])
        largest_eigenvalue_old = 0.0
        v = torch.randn_like(x)
        x_ = x.detach().requires_grad_(True)

        for _ in range(num_iterations):
            with torch.enable_grad():
                Hv, = torch.autograd.grad(grad_func(x_), x_, v, allow_unused=True)
            v = Hv / torch.norm(Hv)
            largest_eigenvalue = v @ Hv
            # Check for convergence
            if torch.abs(largest_eigenvalue - largest_eigenvalue_old) < tol:
                break

            largest_eigenvalue_old = largest_eigenvalue.clone()

        x_.detach_().requires_grad_(True)
        v.detach_().requires_grad_(False)


        # Save input and eigenvector for backward pass
        # x_ = x.detach().requires_grad_(True)
        # v_ = v.detach().requires_grad_(False)
        ctx.save_for_backward(x_, v)
        ctx.grad_func = grad_func

        return largest_eigenvalue

    @staticmethod
    def backward(ctx, grad_output):
        x, v = ctx.saved_tensors
        grad_func = ctx.grad_func

        # Compute gradient of the largest eigenvalue with respect to x
        with torch.enable_grad():
            d2fv = torch.autograd.grad(grad_func(x), x, v, allow_unused=True)[0]
            if d2fv.requires_grad:
                vd3fv = torch.autograd.grad(d2fv, x, v, allow_unused=True)[0]
                grad = vd3fv
            else:
                grad = torch.zeros_like(x)

        return None, grad_output * grad, None, None

def Lambda(f, x, num_iterations=10, tol=1e-6) -> torch.Tensor:
    return LargestEigenvalueFunction.apply(f, x, num_iterations, tol) # type: ignore


def power_iteration(grad_func, x, num_iterations=10):
    # Initialize a random vector
    v = torch.randn_like(x)
    x_ = x.detach().requires_grad_(True)
    
    for _ in range(num_iterations):
        # Compute the gradient of the function at x
        # grad = torch.autograd.grad(func(x), x, create_graph=True)[0]

        # Compute the Jacobian
        # Compute the Jacobian-vector product (JVP) of the gradient with respect to v
        # Hv = torch.autograd.grad(grad, x, v, retain_graph=True)[0]
        # Hv = torch.func.jvp(grad_func, x, v)[1]
        # Hv = torch.autograd.functional.jvp(grad_func, x, v)[1]
        with torch.enable_grad():
            Hv, = torch.autograd.grad(grad_func(x_), x_, v, retain_graph=True, allow_unused=True)
        #     Hv, = torch.func.jvp()
        # Hv = torch.func.jvp(grad_func, (x_,), (v,))[1]

        # Update the estimate of the largest eigenvector
        v = Hv / torch.norm(Hv)

        # Compute the largest eigenvalue using the Rayleigh quotient
    # largest_eigenvalue = (v @ Hv).item()
    largest_eigenvalue = v @ Hv

    return largest_eigenvalue


if __name__ == '__main__':
    import torch

    def f(x):
        return x * torch.tensor([1.0, 2.0, 3.0])


    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    L = Lambda(f, x)
    L.backward()

    print(L, x.grad)