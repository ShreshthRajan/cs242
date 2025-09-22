import matplotlib.pyplot as plt
import numpy as np

# Matrix sizes
N = [16, 32, 64, 128, 256, 512, 1024]

# Runtime data (nanoseconds)
inner_product = [1958, 19049, 183612, 1855381, 18967431, 149495326, 1294175983]
outer_product = [891, 4466, 29599, 209895, 1903705, 29642350, 231868297]

plt.figure(figsize=(10, 6))
plt.plot(N, inner_product, 'b-o', label='Inner Product MMM')
plt.plot(N, outer_product, 'r-o', label='Outer Product MMM')

plt.xlabel('Matrix Dimension (N)')
plt.ylabel('Runtime (nanoseconds)')
plt.title('Matrix Multiplication Runtime Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
# Linear scales as requested

plt.tight_layout()
plt.savefig('mmm_runtime_plot.png', dpi=300, bbox_inches='tight')
plt.show()