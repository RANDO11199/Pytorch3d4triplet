import torch
import math
import numpy as np
from torch_harmonics import *
import torch.nn as nn
from numba import jit
'''
code taken and adapted from pyredner
'''
# modified, from here https://github.com/facebookresearch/pytorch3d/compare/main...ostapagon:pytorch3d:main
# Code adapted from "Spherical Harmonic Lighting: The Gritty Details", Robin Green
# [1] http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf

###Original Authur⬆ 

# TODO: 1. Batch Ops, 2. Make a CUDA
# Write a numba first, and cuda someday later

# see also:【1】 https://learn.microsoft.com/en-us/windows/win32/direct3d9/spherical-environment-mapping
# Tiger book p283

class SphericalHarmonics:
    def __init__(self, envMapResolution, device):
        self.device = device
        self.setEnvironmentMapResolution(envMapResolution)

    def setEnvironmentMapResolution(self, res):
        self.resolution = res
        uv = np.mgrid[0:res[1], 0:res[0]].astype(np.float32)
        self.theta = torch.from_numpy((math.pi / res[1]) * (uv[1, :, :] + 0.5)).to(self.device)
        self.phi = torch.from_numpy((2 * math.pi / res[0]) * (uv[0, :, :] + 0.5)).to(self.device)

    def smoothSH(self, coeffs, window=6):
        ''' multiply (convolve in sptial domain) the coefficients with a low pass filter.
        Following the recommendation in https://www.ppsloan.org/publications/shdering.pdf
        '''
        smoothed_coeffs = torch.zeros_like(coeffs)
        smoothed_coeffs[:, 0] += coeffs[:, 0]
        smoothed_coeffs[:, 1:1 + 3] += \
            coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
        smoothed_coeffs[:, 4:4 + 5] += \
            coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
        smoothed_coeffs[:, 9:9 + 7] += \
            coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
        return smoothed_coeffs


    def associatedLegendrePolynomial(self, l, m, x):
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0
        if l == m:
            return pmm
        pmmp1 = x * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1
        pll = torch.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        return pll

    #[1] P12 ,K
    def normlizeSH(self, l, m):
        return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / \
                         (4 * math.pi * math.factorial(l + m)))
    #[1] p12, SH
    def SH(self, l, m, theta, phi):
        if m == 0:
            return self.normlizeSH(l, m) * self.associatedLegendrePolynomial(l, m, torch.cos(theta))
        elif m > 0:
            return math.sqrt(2.0) * self.normlizeSH(l, m) * \
                   torch.cos(m * phi) * self.associatedLegendrePolynomial(l, m, torch.cos(theta))
        else:
            return math.sqrt(2.0) * self.normlizeSH(l, -m) * \
                   torch.sin(-m * phi) * self.associatedLegendrePolynomial(l, -m, torch.cos(theta))
    
    def toEnvMap(self, shCoeffs, smooth = False):
        '''
        create an environment map from given sh coeffs
        :param shCoeffs: float tensor [n, bands * bands, 3]
        :param smooth: if True, the first 3 bands are smoothed
        :return: environment map tensor [n, resX, resY, 3]
        '''
        assert(shCoeffs.dim() == 3 and shCoeffs.shape[-1] == 3)
        envMaps = torch.zeros( [shCoeffs.shape[0], self.resolution[1], self.resolution[0], 3]).to(shCoeffs.device)
        for i in range(shCoeffs.shape[0]):
            envMap = self.constructEnvMapFromSHCoeffs(shCoeffs[i], smooth)
            envMaps[i] = envMap
        return envMaps


    def constructEnvMapFromSHCoeffs(self, shCoeffs, smooth = False):

        assert (isinstance(shCoeffs, torch.Tensor) and shCoeffs.dim() == 2 and shCoeffs.shape[1] == 3)

        if smooth:
            smoothed_coeffs = self.smoothSH(shCoeffs.transpose(0, 1), 4)
        else:
            smoothed_coeffs =  shCoeffs.transpose(0, 1) #self.smoothSH(shCoeffs.transpose(0, 1), 4) #smooth the first three bands?

        res = self.resolution

        theta = self.theta 
        phi =  self.phi
        result = torch.zeros(res[1], res[0], smoothed_coeffs.shape[0], device=smoothed_coeffs.device)
        bands = int(math.sqrt(smoothed_coeffs.shape[1]))
        i = 0

        for l in range(bands):
            for m in range(-l, l + 1):
                sh_factor = self.SH(l, m, theta, phi)
                result = result + sh_factor.view(sh_factor.shape[0], sh_factor.shape[1], 1) * smoothed_coeffs[:, i]
                i += 1
        result = torch.max(result, torch.zeros(res[1], res[0], smoothed_coeffs.shape[0], device=smoothed_coeffs.device))
        return result
    def constructEnvMapFromSHCoeffsFast(self, shCoeffs, smooth=False):
        assert (isinstance(shCoeffs, torch.Tensor) and shCoeffs.dim() == 2 and shCoeffs.shape[1] == 3)

        if smooth:
            smoothed_coeffs = self.smoothSH(shCoeffs.transpose(0, 1), 4)
        else:
            smoothed_coeffs = shCoeffs.transpose(0, 1)

        res = self.resolution
        theta = self.theta  # shape [H, W]
        phi = self.phi      # shape [H, W]
        
        # Prepare a tensor to store all SH factors
        result = torch.zeros(res[1], res[0], smoothed_coeffs.shape[0], device=smoothed_coeffs.device)
        
        # Compute the number of bands
        bands = int(math.sqrt(smoothed_coeffs.shape[1]))
        
        # Prepare tensors for storing SH factors
        sh_factors = []

        # Vectorized computation of all SH factors without explicit loops
        for l in range(bands):
            m_range = torch.arange(-l, l + 1, device=self.device)
            m_grid = m_range.view(-1, 1, 1)  # Shape [2*l+1, 1, 1]
            
            # Compute cos and sin terms for all m in parallel
            cos_terms = torch.cos(m_grid * phi)  # Shape [2*l+1, H, W]
            sin_terms = torch.sin(-m_grid * phi)  # Shape [2*l+1, H, W]
            
            # Compute Associated Legendre Polynomials for all m in parallel
            legendre_poly = torch.stack([self.associatedLegendrePolynomial(l, m, torch.cos(theta)) for m in m_range])  # Shape [2*l+1, H, W]
            
            # Compute normalization factors in parallel
            normalization_factors = torch.tensor([self.normlizeSH(l, m.item()) for m in m_range], device=self.device).view(-1, 1, 1)  # Shape [2*l+1, 1, 1]
            
            # Compute the SH factors for all m in parallel
            sh_factors_m = normalization_factors * legendre_poly
            sh_factors_m[1:] *= math.sqrt(2.0)  # Apply the sqrt(2) scaling for m != 0
            sh_factors_m[1:] *= cos_terms[1:]  # Apply cos(m * phi) for m > 0
            sh_factors_m[1:] += sin_terms[1:]  # Apply sin(-m * phi) for m < 0
            
            sh_factors.append(sh_factors_m)
        
        # Concatenate SH factors across all bands
        sh_factors = torch.cat(sh_factors, dim=0)  # Shape [num_SH, H, W]

        # Prepare coefficients tensor for broadcasting
        coeffs = smoothed_coeffs.view(1, 1, smoothed_coeffs.shape[0], -1)  # Shape [1, 1, C, num_SH]

        # Compute the result by broadcasting and summing over SH dimension
        result = torch.einsum('ijk,ijlm->ijl', sh_factors, coeffs)

        # Apply the max operation in a differentiable manner
        result = torch.max(result, torch.zeros_like(result, device=result.device))
        return result


class SpectralModel(nn.Module):
    def __init__(self, n_modes, n_theta, n_lambda,batch=1):
        super().__init__()
        sht = RealSHT(n_theta, n_lambda, lmax=n_modes, mmax=n_modes+1, grid="equiangular",norm='schmidt')
        coeffs = sht(torch.ones(1,3,n_theta, n_lambda)*0.5).to(dtype=torch.complex128)
        # self.coeffs = nn.Parameter(torch.randn(batch,3,n_modes, n_modes+1, dtype=torch.complex128))
        self.coeffs = nn.Parameter(coeffs)
        # del sht,coeffs
        self.isht = InverseRealSHT(n_theta, n_lambda, lmax=n_modes, mmax=n_modes+1, grid="equiangular",norm='schmidt').to('cuda')
    def forward(self):
        return self.isht(self.coeffs)


import math
import numpy as np
import torch
from numba import cuda

class SphericalHarmonics_CUDA:
    def __init__(self, envMapResolution, device):
        self.device = device
        self.setEnvironmentMapResolution(envMapResolution)

    def setEnvironmentMapResolution(self, res):
        self.resolution = res
        uv = np.mgrid[0:res[1], 0:res[0]].astype(np.float32)
        self.theta = torch.from_numpy((math.pi / res[1]) * (uv[1, :, :] + 0.5)).to(self.device)
        self.phi = torch.from_numpy((2 * math.pi / res[0]) * (uv[0, :, :] + 0.5)).to(self.device)

    def smoothSH(self, coeffs, window=6):
        ''' multiply (convolve in sptial domain) the coefficients with a low pass filter.'''
        smoothed_coeffs = torch.zeros_like(coeffs)
        smoothed_coeffs[:, 0] += coeffs[:, 0]
        smoothed_coeffs[:, 1:1 + 3] += coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
        smoothed_coeffs[:, 4:4 + 5] += coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
        smoothed_coeffs[:, 9:9 + 7] += coeffs[:, 9:9 + 7] * math.pow(math.sin(math.pi * 3.0 / window) / (math.pi * 3.0 / window), 4.0)
        return smoothed_coeffs

    def associatedLegendrePolynomial(self, l, m, x):
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0
        if l == m:
            return pmm
        pmmp1 = x * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1
        pll = torch.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        return pll

    def normlizeSH(self, l, m):
        return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / (4 * math.pi * math.factorial(l + m)))

    def SH(self, l, m, theta, phi):
        cos_theta = torch.cos(theta)
        if m == 0:
            return self.normlizeSH(l, m) * self.associatedLegendrePolynomial(l, m, cos_theta)
        elif m > 0:
            return math.sqrt(2.0) * self.normlizeSH(l, m) * torch.cos(m * phi) * self.associatedLegendrePolynomial(l, m, cos_theta)
        else:
            return math.sqrt(2.0) * self.normlizeSH(l, -m) * torch.sin(-m * phi) * self.associatedLegendrePolynomial(l, -m, cos_theta)

    def constructEnvMapFromSHCoeffs(self, shCoeffs, smooth=False):
        assert (isinstance(shCoeffs, torch.Tensor) and shCoeffs.dim() == 2 and shCoeffs.shape[1] == 3)

        if smooth:
            smoothed_coeffs = self.smoothSH(shCoeffs.transpose(0, 1), 4)
        else:
            smoothed_coeffs = shCoeffs.transpose(0, 1)

        res = self.resolution
        theta = self.theta
        phi = self.phi

        result = torch.zeros(res[1], res[0], smoothed_coeffs.shape[0], device=smoothed_coeffs.device)
        bands = int(math.sqrt(smoothed_coeffs.shape[1]))

        # Flattened number of (l, m) pairs
        num_elements = sum(2 * l + 1 for l in range(bands))

        # Allocate result array on the GPU
        d_result = cuda.to_device(result.cpu().numpy())
        d_smoothed_coeffs = cuda.to_device(smoothed_coeffs.cpu().numpy())
        d_theta = cuda.to_device(theta.cpu().numpy())
        d_phi = cuda.to_device(phi.cpu().numpy())

        # Define block and grid dimensions
        threads_per_block = 256
        blocks_per_grid = (num_elements + threads_per_block - 1) // threads_per_block

        # Launch CUDA kernel
        self.compute_sh_kernel[blocks_per_grid, threads_per_block](d_result, d_smoothed_coeffs, d_theta, d_phi, bands)

        # Copy results back to the host
        result = torch.from_numpy(d_result.copy_to_host())

        # Compute the max operation
        result = torch.max(result, torch.zeros_like(result, device=result.device))
        return result

    @staticmethod
    @cuda.jit
    def compute_sh_kernel(result, smoothed_coeffs, theta, phi, bands):
        idx = cuda.grid(1)

        # Determine l and m based on the flat index
        l = 0
        offset = idx
        for i in range(bands):
            num_m_values = 2 * i + 1
            if offset < num_m_values:
                l = i
                break
            offset -= num_m_values
        m = offset - l

        # Ensure valid indices
        if l < bands and m >= -l and m <= l:
            # Compute Spherical Harmonics (SH) factor
            sh_factor = SphericalHarmonics.SH(l, m, theta, phi)

            # Perform the multiplication and addition
            i = l * (2 * bands - 1) + (m + l)  # Calculate unique index for i
            for x in range(result.shape[0]):
                for y in range(result.shape[1]):
                    for z in range(smoothed_coeffs.shape[0]):
                        result[x, y, z] += sh_factor[x, y] * smoothed_coeffs[z, i]

    def toEnvMap(self, shCoeffs, smooth=False):
        assert(shCoeffs.dim() == 3 and shCoeffs.shape[-1] == 3)
        envMaps = torch.zeros([shCoeffs.shape[0], self.resolution[1], self.resolution[0], 3]).to(shCoeffs.device)
        for i in range(shCoeffs.shape[0]):
            envMap = self.constructEnvMapFromSHCoeffs(shCoeffs[i], smooth)
            envMaps[i] = envMap
        return envMaps

# Example usage:
sh = SphericalHarmonics((512, 512), "cuda")
sh_coeffs = torch.randn((10, 9, 3), device="cuda")
result = sh.toEnvMap(sh_coeffs)
print(result.shape)