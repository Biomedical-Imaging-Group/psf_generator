"""Minimalist example: Fit Zernike coefficients using gradient descent."""
import torch
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, 'src')

from psf_generator.propagators.scalar_cartesian_propagator import ScalarCartesianPropagator

# Settings
n_pix = 128
n_zernike = 10
learning_rate = 0.5
n_iterations = 200

print("=== Zernike Coefficient Fitting Demo ===\n")

# Create target PSF (no aberrations)
print("Creating target PSF (no aberrations)...")
target_coeffs = torch.zeros(n_zernike)
target_prop = ScalarCartesianPropagator(
    n_pix_pupil=n_pix,
    n_pix_psf=n_pix,
    zernike_coefficients=target_coeffs
)
target_psf = torch.abs(target_prop.compute_focus_field()) ** 2
print(f"Target coefficients: {target_coeffs.numpy()}\n")

# Initialize with random aberrations
print("Initializing with random Zernike aberrations...")
initial_coeffs = torch.randn(n_zernike) * 1
initial_coeffs[0] = 0  # Keep piston at 0
initial_coeffs.requires_grad = True
print(f"Initial coefficients: {initial_coeffs.detach().numpy()}\n")

# Create propagator for optimization
prop = ScalarCartesianPropagator(
    n_pix_pupil=n_pix,
    n_pix_psf=n_pix,
    zernike_coefficients=initial_coeffs.detach()
)

# Gradient descent
print("Starting gradient descent...\n")
losses = []
coeffs_history = [initial_coeffs.detach().clone()]

optimizer = torch.optim.Adam([initial_coeffs], lr=learning_rate)

for iteration in range(n_iterations):
    optimizer.zero_grad()

    # Update propagator with current coefficients
    prop.update_zernike_coefficients(initial_coeffs)

    # Compute PSF
    field = prop.compute_focus_field()
    psf = torch.abs(field) ** 2

    # Loss: MSE between PSFs
    loss = torch.mean((psf - target_psf) ** 2)

    # Backward pass
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    coeffs_history.append(initial_coeffs.detach().clone())

    if (iteration + 1) % 20 == 0:
        print(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss.item():.6e}")

print(f"\n✓ Optimization complete!\n")

# Results
final_coeffs = initial_coeffs.detach()
print("=== Results ===")
print(f"Target coefficients:  {target_coeffs.numpy()}")
print(f"Final coefficients:   {final_coeffs.numpy()}")
print(f"Difference (L2 norm): {torch.norm(final_coeffs - target_coeffs).item():.6f}")
print(f"Final loss:           {losses[-1]:.6e}\n")

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Target PSF
axes[0, 0].imshow(target_psf[0, 0].detach().cpu().numpy(), cmap='hot')
axes[0, 0].set_title('Target PSF (no aberrations)')
axes[0, 0].axis('off')

# Initial PSF
prop.update_zernike_coefficients(coeffs_history[0])
initial_psf = torch.abs(prop.compute_focus_field()) ** 2
axes[0, 1].imshow(initial_psf[0, 0].detach().cpu().numpy(), cmap='hot')
axes[0, 1].set_title('Initial PSF (random aberrations)')
axes[0, 1].axis('off')

# Final PSF
prop.update_zernike_coefficients(final_coeffs)
final_psf = torch.abs(prop.compute_focus_field()) ** 2
axes[0, 2].imshow(final_psf[0, 0].detach().cpu().numpy(), cmap='hot')
axes[0, 2].set_title('Final PSF (after fitting)')
axes[0, 2].axis('off')

# Loss curve
axes[1, 0].semilogy(losses)
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Loss (MSE)')
axes[1, 0].set_title('Loss Curve')
axes[1, 0].grid(True, alpha=0.3)

# Coefficient convergence
coeffs_array = torch.stack(coeffs_history).numpy()
for i in range(min(5, n_zernike)):
    axes[1, 1].plot(coeffs_array[:, i], label=f'Z{i}')
axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Coefficient Value')
axes[1, 1].set_title('Coefficient Convergence')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Bar plot: initial, final, target
x = range(n_zernike)
width = 0.25
axes[1, 2].bar([xi - width for xi in x], coeffs_history[0].numpy(), width=width,
               label='Initial', alpha=0.7, color='C0')
axes[1, 2].bar([xi for xi in x], final_coeffs.numpy(), width=width,
               label='Final', alpha=0.7, color='C1')
axes[1, 2].bar([xi + width for xi in x], target_coeffs.numpy(), width=width,
               label='Target', alpha=0.7, color='C2')
axes[1, 2].set_xlabel('Zernike Index')
axes[1, 2].set_ylabel('Coefficient Value')
axes[1, 2].set_title('Zernike Coefficients Comparison')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zernike_fitting_demo.png', dpi=150, bbox_inches='tight')
print("✓ Results saved to 'zernike_fitting_demo.png'")
plt.show()
