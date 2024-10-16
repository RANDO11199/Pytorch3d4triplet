import torch
from torch.autograd import Function
torch.set_default_device('cuda')
from einops import einsum
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792,-1.0925484305920792,0.31539156525252005,-1.0925484305920792,0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
SH_C4 = (2.5033429417967,-1.77013076977993,0.9461746957575601,-0.6690465435572892,0.10578554691520431,-0.6690465435572892,0.47308734787878004,-1.77013076977993,0.6258357354491761)
SH_C5 = (-0.65638205684017,8.30264925952416,0.00931882475114763,0.0913054625709205,-0.45294665119569694,0.1169503224534236,-0.45294665119569694,2.396768392486662,0.4892382994352504,2.07566231488104,-0.65638205684017)
# Make a theta phi version for transformation of cubemap
def RGB2SH(rgb):
    return (rgb - 0.5) / SH_C0

def SH2RGB(sh):
    return sh * SH_C0 + 0.5
def dnormvdv3d(unnormalized_v:torch.Tensor,
                direction_v_x : torch.Tensor,
                direction_v_y : torch.Tensor,
                direction_v_z : torch.Tensor):
    sum2 = unnormalized_v[...,0] * unnormalized_v[...,0] + unnormalized_v[...,1] * unnormalized_v[...,1]+\
           unnormalized_v[...,2] * unnormalized_v[...,2] # (batch, vert,1)
    invsum32 = 1.0/torch.sqrt(sum2 * sum2 * sum2)

    dnormvdv = torch.zeros_like(unnormalized_v)
    dnormvdv[...,0] = ((+sum2 - unnormalized_v[...,0] * unnormalized_v[...,0]) * direction_v_x
                       - unnormalized_v[...,1] * unnormalized_v[...,0] * direction_v_y 
                       - unnormalized_v[...,2] * unnormalized_v[...,0] * direction_v_z) * invsum32
    
    dnormvdv[...,1] = (-unnormalized_v[...,0] * unnormalized_v[...,1] * direction_v_x + 
                       (sum2 - unnormalized_v[...,1] * unnormalized_v[...,1]) * direction_v_y -
                        unnormalized_v[...,2] * unnormalized_v[...,1] * direction_v_z ) * invsum32
    
    dnormvdv[...,2] = (-unnormalized_v[...,0] * unnormalized_v[...,2] * direction_v_x -
                       unnormalized_v[...,1] * unnormalized_v[...,2] * direction_v_y + 
                       (sum2 - unnormalized_v[...,2] * unnormalized_v[...,2]) * direction_v_z)*invsum32
    return dnormvdv

# From https://github.com/cheind/torch-spherical-harmonics and GS3D
# Modified part of the expression aesthetically, it is numerically close (and analytically the same) if input is normalized
class SphereHarmonic(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                position:torch.Tensor,
                degree:int,
                max_coeffs:int,
                camera_positions:torch.FloatTensor,
                SphereHarmonics_Coeffs:torch.FloatTensor, # (batch size/mesh, vertex,rgb, max_coeffs)
                compensation = 0.5
                )-> torch.Tensor:
        dir_orig = position - camera_positions
        dir = torch.nn.functional.normalize(dir_orig,p=2,dim=-1,eps=1e-12)
        result = SH_C0 * SphereHarmonics_Coeffs[...,0]
        if (degree>0):
            x = dir[...,0].unsqueeze(dim=-1)
            y = dir[...,1].unsqueeze(dim=-1)
            z = dir[...,2].unsqueeze(dim=-1)
            result = result - \
                              SH_C1 * y * SphereHarmonics_Coeffs[...,1] +\
                              SH_C1 * z * SphereHarmonics_Coeffs[...,2] -\
                              SH_C1 * x * SphereHarmonics_Coeffs[...,3]
            if (degree > 1):
                xx = x*x
                yy = y*y
                zz = z*z
                xy = x*y
                yz = y*z
                xz = x*z
                result = result + \
                                SH_C2[0] * xy * SphereHarmonics_Coeffs[...,4] + \
                                SH_C2[1] * yz * SphereHarmonics_Coeffs[...,5] + \
                                SH_C2[2] * (2.0 * zz - xx - yy) * SphereHarmonics_Coeffs[...,6] + \
                                SH_C2[3] * xz * SphereHarmonics_Coeffs[...,7] + \
                                SH_C2[4] * (xx - yy) * SphereHarmonics_Coeffs[...,8]
                if (degree > 2):
                    result = result + \
                                      SH_C3[0] * y * (3.0 * xx - yy) * SphereHarmonics_Coeffs[...,9] + \
                                      SH_C3[1] * xy * z *SphereHarmonics_Coeffs[...,10] + \
                                      SH_C3[2] * y * (4.0 * zz - xx - yy) * SphereHarmonics_Coeffs[...,11] + \
                                      SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * SphereHarmonics_Coeffs[...,12] +\
                                      SH_C3[4] * x * (4.0 * zz - xx - yy) * SphereHarmonics_Coeffs[...,13] + \
                                      SH_C3[5] * z * (xx - yy) * SphereHarmonics_Coeffs[...,14] + \
                                      SH_C3[6] * x * (xx - 3.0 * yy) * SphereHarmonics_Coeffs[...,15]
                if (degree > 3):
                    x4 = xx*xx
                    y4 = yy * yy
                    z4 = zz * zz
                    result = result + \
                                      SH_C4[0] * xy * (xx - yy)* SphereHarmonics_Coeffs[...,16] +\
                                      SH_C4[1] * yz * (3.0 * xx -yy)* SphereHarmonics_Coeffs[...,17] +\
                                      SH_C4[2] * xy * (6*zz - xx - yy)* SphereHarmonics_Coeffs[...,18] +\
                                      SH_C4[3] * yz * (4 * zz - 3 * xx - 3 * yy)* SphereHarmonics_Coeffs[...,19] +\
                                      SH_C4[4] * ( 5 * z4 - 30*zz*(xx + yy) + 3) * SphereHarmonics_Coeffs[...,20] +\
                                      SH_C4[5] * xz * (4 * zz - 3 * xx - 3 * yy)* SphereHarmonics_Coeffs[...,21] +\
                                      SH_C4[6] * (xx - yy)*(6*zz - xx - yy)* SphereHarmonics_Coeffs[...,22]+\
                                      SH_C4[7] * xz * (xx - 3 * yy)* SphereHarmonics_Coeffs[...,23] +\
                                      SH_C4[8] * (x4 + y4 - 6 * xx * yy)* SphereHarmonics_Coeffs[...,24]
                if (degree > 4):
                    result = result +\
                                      SH_C5[0] *  y * (-10.0 * xx * yy + 5.0 * x4 + y4)* SphereHarmonics_Coeffs[...,25] +\
                                      SH_C5[1] * xy * z * (xx - yy)* SphereHarmonics_Coeffs[...,26] +\
                                      SH_C5[2] * y * (52.5 - 472.5 * zz) * (3.0 * xx - yy)* SphereHarmonics_Coeffs[...,27] +\
                                      SH_C5[3] * xy * (3.0 * z * (52.5 * zz - 7.5) - 30.0 * z)* SphereHarmonics_Coeffs[...,28] +\
                                      SH_C5[4] * y * (8 - 28 * (xx + yy)  + 21*( x4 + y4 + 2*xx*yy))* SphereHarmonics_Coeffs[...,29] +\
                                      SH_C5[5] * z * (8 - 56 * (xx + yy) + 63*(x4 + 2 * xx * yy +  y4) ) * SphereHarmonics_Coeffs[...,30] +\
                                      SH_C5[6] * x * (8 - 28 * (xx + yy) + 21 * (x4 + 2*xx*yy + y4))* SphereHarmonics_Coeffs[...,31] +\
                                      SH_C5[7] * z * (yy - xx)* (-2 + 3*xx + 3*yy)* SphereHarmonics_Coeffs[...,32] +\
                                      SH_C5[8] * x * (8*zz - xx - yy)*(xx - 3 * yy)* SphereHarmonics_Coeffs[...,33] +\
                                      SH_C5[9]  * z * (-6.0 * xx * zz + x4 + y4)* SphereHarmonics_Coeffs[...,34]+\
                                      SH_C5[10] * x * (-10.0 * xx * yy + x4 + 5.0 * y4)* SphereHarmonics_Coeffs[...,35]                                                 
        result += compensation
        ctx.sh_config = (max_coeffs,degree)
        clamped = result < 0
        ctx.save_for_backward(dir_orig, SphereHarmonics_Coeffs, clamped)
        return torch.nn.functional.relu(result)
    @staticmethod
    def backward(ctx,dL_dcolor):
        
        dir_orig, SphereHarmonics_Coeffs, clamped = ctx.saved_tensors 
        # position = position
        SphereHarmonics_Coeffs = SphereHarmonics_Coeffs
        # camera_positions = camera_positions
        clamped = clamped
        max_coeffs, degree = ctx.sh_config
    
        # dir_orig = position 
        # dir_orig = dir_orig - camera_positions
        # dir = dir - camera_positions 
        dir = torch.nn.functional.normalize(dir_orig,p=2,dim=-1,eps=1e-12)
        # dir = dir - camera_positions 
        x = dir[...,0].unsqueeze(dim=-1)
        y = dir[...,1].unsqueeze(dim=-1)
        z = dir[...,2].unsqueeze(dim=-1)

        dL_dRGB = dL_dcolor# (Batch size , vertex, RGB)

        mask_clamped = torch.where(clamped, 0, 1)
        
        dL_dRGB = dL_dRGB * mask_clamped
        dL_dRGB = dL_dRGB
        dRGBdsh0 = SH_C0

        dL_dsh = torch.zeros( dL_dRGB.shape[0], dL_dRGB.shape[1], 6 , max_coeffs,device='cuda')

        dL_dsh[...,0] = dL_dRGB * dRGBdsh0
        dRGBdx = torch.zeros(SphereHarmonics_Coeffs.shape[0],SphereHarmonics_Coeffs.shape[1],6,1,device='cuda')
        dRGBdy = torch.zeros(SphereHarmonics_Coeffs.shape[0],SphereHarmonics_Coeffs.shape[1],6,1,device='cuda')
        dRGBdz = torch.zeros(SphereHarmonics_Coeffs.shape[0],SphereHarmonics_Coeffs.shape[1],6,1,device='cuda')

        if (degree > 0):
            dRGBdsh1 = (-SH_C1 * y)
            dRGBdsh2 = (SH_C1 * z)
            dRGBdsh3 = (-SH_C1 * x)
            dL_dsh[...,1] = dRGBdsh1 * dL_dRGB
            dL_dsh[...,2] = dRGBdsh2 * dL_dRGB
            dL_dsh[...,3] = dRGBdsh3 * dL_dRGB

            dRGBdx = -SH_C1 * SphereHarmonics_Coeffs[...,3]
            dRGBdy = -SH_C1 * SphereHarmonics_Coeffs[...,1]
            dRGBdz = SH_C1 * SphereHarmonics_Coeffs[...,2]

            if degree > 1:
                xx = x * x
                yy = y * y
                
                zz = z * z 
                xy = x * y 
                yz = y * z 
                xz = x * z 
                
                dRGBdsh4 = (SH_C2[0] * xy )
                dRGBdsh5 = (SH_C2[1] * yz)
                dRGBdsh6 = (SH_C2[2] * (2.0 * zz - xx - yy))
                dRGBdsh7 = (SH_C2[3] * xz)
                dRGBdsh8 = (SH_C2[4] * (xx - yy))
                
                dL_dsh[...,4] = dRGBdsh4 * dL_dRGB
                dL_dsh[...,5] = dRGBdsh5 * dL_dRGB
                dL_dsh[...,6] = dRGBdsh6 * dL_dRGB
                dL_dsh[...,7] = dRGBdsh7 * dL_dRGB
                dL_dsh[...,8] = dRGBdsh8 * dL_dRGB
                
                dRGBdx += SH_C2[0] * y * SphereHarmonics_Coeffs[...,4] +  \
                        SH_C2[2] * 2. * -x * SphereHarmonics_Coeffs[...,6] + \
                        SH_C2[3] * z * SphereHarmonics_Coeffs[...,7] + \
                        SH_C2[4] * 2.0 * x * SphereHarmonics_Coeffs[...,8]
                        
                dRGBdy += SH_C2[0] * x * SphereHarmonics_Coeffs[...,4] + \
                        SH_C2[1] * z * SphereHarmonics_Coeffs[...,5] + \
                        SH_C2[2] * 2.0 * -y * SphereHarmonics_Coeffs[...,6] + \
                        SH_C2[4] * 2.0 * -y *SphereHarmonics_Coeffs[...,8]
                
                dRGBdz += SH_C2[1] * y *SphereHarmonics_Coeffs[...,5] + \
                        SH_C2[2] * 2.0 * 2.0 * z * SphereHarmonics_Coeffs[...,6] + \
                        SH_C2[3] * x * SphereHarmonics_Coeffs[...,7]
                
                if degree >2 :
                    dRGBdsh9 = SH_C3[0] * y * (3.0 * xx - yy)
                    dRGBdsh10 = SH_C3[1] * xy * z 
                    dRGBdsh11 = SH_C3[2] * y * (4.0 * zz - xx - yy)
                    dRGBdsh12 = SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                    dRGBdsh13 = SH_C3[4] * x * (4.0 * zz - xx - yy)
                    dRGBdsh14 = SH_C3[5] * z * (xx - yy)
                    dRGBdsh15 = SH_C3[6] * x * (xx - 3.0 *yy)
                    dL_dsh[...,9] = dRGBdsh9 * dL_dRGB
                    dL_dsh[...,10] = dRGBdsh10 * dL_dRGB
                    dL_dsh[...,11] = dRGBdsh11 * dL_dRGB
                    dL_dsh[...,12] = dRGBdsh12 * dL_dRGB
                    dL_dsh[...,13] = dRGBdsh13 * dL_dRGB
                    dL_dsh[...,14] = dRGBdsh14 * dL_dRGB
                    dL_dsh[...,15] = dRGBdsh15 * dL_dRGB

                    dRGBdx += SH_C3[0] * SphereHarmonics_Coeffs[...,9] * 3.0 * 2.0 * xy +\
                            SH_C3[1] * SphereHarmonics_Coeffs[...,10] * yz + \
                            SH_C3[2] * SphereHarmonics_Coeffs[...,11] * -2.0 * xy + \
                            SH_C3[3] * SphereHarmonics_Coeffs[...,12] * -3.0 * 2.0 * xz + \
                            SH_C3[4] * SphereHarmonics_Coeffs[...,13] * (-3.0 * xx + 4.0 * zz - yy) + \
                            SH_C3[5] * SphereHarmonics_Coeffs[...,14] * 2.0 * xz + \
                            SH_C3[6] * SphereHarmonics_Coeffs[...,15] * 3.0 * (xx - yy)
                            
                    
                    dRGBdy += SH_C3[0] * SphereHarmonics_Coeffs[...,9] * 3.0 * (xx - yy) +\
                            SH_C3[1] * SphereHarmonics_Coeffs[...,10] * xz + \
                            SH_C3[2] * SphereHarmonics_Coeffs[...,11] * (-3.0 * yy + 4.0 * zz - xx) +\
                            SH_C3[3] * SphereHarmonics_Coeffs[...,12] * -3.0 * 2.0 * yz +\
                            SH_C3[4] * SphereHarmonics_Coeffs[...,13] * -2.0 * xy + \
                            SH_C3[5] * SphereHarmonics_Coeffs[...,14] * -2.0 * yz + \
                            SH_C3[6] * SphereHarmonics_Coeffs[...,15] * -3.0 * 2.0 * xy
                    
                    
                    dRGBdz += SH_C3[1] * SphereHarmonics_Coeffs[...,10] * xy + \
                            SH_C3[2] * SphereHarmonics_Coeffs[...,11] * 4.0 * 2.0 * yz +\
                            SH_C3[3] * SphereHarmonics_Coeffs[...,12] * 3.0 * (2.0 * zz - xx - yy) +\
                            SH_C3[4] * SphereHarmonics_Coeffs[...,13] * 4.0 * 2.0 * xz + \
                            SH_C3[5] * SphereHarmonics_Coeffs[...,14] * (xx - yy)
                if degree >3 :
                    y3 = y*y*y
                    z3 = z*z*z
                    x3 = x*x*x
                    z4 = z*z*z*z
                    x4 = x*x*x*x
                    y4 = y*y*y*y
                    dRGBdsh16 = SH_C4[0] * xy * (xx - yy)
                    dRGBdsh17 = SH_C4[1] * yz * (3.0 * xx -yy)
                    dRGBdsh18 = SH_C4[2] * xy * (6*zz - xx - yy)
                    dRGBdsh19 = SH_C4[3] * yz * (4 * zz - 3 * xx - 3 * yy)
                    dRGBdsh20 = SH_C4[4] * ( 5 * z4 - 30*zz*(xx + yy) + 3)
                    dRGBdsh21 = SH_C4[5] * xz * (4 * zz - 3 * xx - 3 * yy)
                    dRGBdsh22 = SH_C4[6] * (xx - yy)*(6*zz - xx - yy)
                    dRGBdsh23 = SH_C4[7] * xz * (xx - 3 * yy)
                    dRGBdsh24 = SH_C4[8] * (x4 + y4 - 6 * xx * yy)

                    dL_dsh[...,16] = dRGBdsh16 * dL_dRGB
                    dL_dsh[...,17] = dRGBdsh17 * dL_dRGB
                    dL_dsh[...,18] = dRGBdsh18 * dL_dRGB
                    dL_dsh[...,19] = dRGBdsh19 * dL_dRGB
                    dL_dsh[...,20] = dRGBdsh20 * dL_dRGB
                    dL_dsh[...,21] = dRGBdsh21 * dL_dRGB
                    dL_dsh[...,22] = dRGBdsh22 * dL_dRGB
                    dL_dsh[...,23] = dRGBdsh23 * dL_dRGB
                    dL_dsh[...,24] = dRGBdsh24 * dL_dRGB

                    dRGBdx += SH_C4[0] * SphereHarmonics_Coeffs[...,16] * (3.*xx*y - y3) +\
                            SH_C4[1] * SphereHarmonics_Coeffs[...,17] * x*y*z * 6. + \
                            SH_C4[2] * SphereHarmonics_Coeffs[...,18] * (6*y*zz - 3*xx*y - y3) + \
                            SH_C4[3] * SphereHarmonics_Coeffs[...,19] * -3.0 * 2.0 * xz*y + \
                            SH_C4[4] * SphereHarmonics_Coeffs[...,20] * (-60.*zz*x) + \
                            SH_C4[5] * SphereHarmonics_Coeffs[...,21] * (4*z3 - 9 *xx*z - 3*z*yy) + \
                            SH_C4[6] * SphereHarmonics_Coeffs[...,22] * (12*x*zz - 4*x3 )+\
                            SH_C4[7] * SphereHarmonics_Coeffs[...,23] * 3.0 * z * (xx - yy)+\
                            SH_C4[8] * SphereHarmonics_Coeffs[...,24] * (4*x3 - 12*x*yy)
                    dRGBdy += SH_C4[0] * SphereHarmonics_Coeffs[...,16] * (x3 - 3*x*yy) +\
                            SH_C4[1] * SphereHarmonics_Coeffs[...,17] * 3.* (xx*z - yy*z) + \
                            SH_C4[2] * SphereHarmonics_Coeffs[...,18] * (6*x*zz - x3 - 3.*x*yy) + \
                            SH_C4[3] * SphereHarmonics_Coeffs[...,19] * (4.*z3 - 3.*xx*z - 9.*yy*z) + \
                            SH_C4[4] * SphereHarmonics_Coeffs[...,20] * (-60. * zz * y) + \
                            SH_C4[5] * SphereHarmonics_Coeffs[...,21] * (-6. * x*y*z) + \
                            SH_C4[6] * SphereHarmonics_Coeffs[...,22] * (4*y3 - 12. * y * zz)+\
                            SH_C4[7] * SphereHarmonics_Coeffs[...,23] * (-6.*x*y*z)+\
                            SH_C4[8] * SphereHarmonics_Coeffs[...,24] * (4*y3 - 12*y*xx)
                    
                    dRGBdz += SH_C4[0] * SphereHarmonics_Coeffs[...,16] * (0.) +\
                            SH_C4[1] * SphereHarmonics_Coeffs[...,17] * (3*xx*y - y3) + \
                            SH_C4[2] * SphereHarmonics_Coeffs[...,18] * (12 * xy*z) + \
                            SH_C4[3] * SphereHarmonics_Coeffs[...,19] * (12*y*zz - 3*xx*y - 3 * y3) + \
                            SH_C4[4] * SphereHarmonics_Coeffs[...,20] * (20*z3-60*z*xx-60*yy*z) + \
                            SH_C4[5] * SphereHarmonics_Coeffs[...,21] * (12*x*zz - 3*x3 - 3 * x*yy) + \
                            SH_C4[6] * SphereHarmonics_Coeffs[...,22] * 12.*z*(xx-yy)+\
                            SH_C4[7] * SphereHarmonics_Coeffs[...,23] * (x3 - 3 * yy *x)+\
                            SH_C4[8] * SphereHarmonics_Coeffs[...,24] * (0.)
                if degree >4 :
                    dRGBdsh25 = SH_C5[0] *  y * (-10.0 * xx * yy + 5.0 * x4 + y4)
                    dRGBdsh26 = SH_C5[1] * xy * z * (xx - yy)
                    dRGBdsh27 = SH_C5[2] * y * (52.5 - 472.5 * zz) * (3.0 * xx - yy)
                    dRGBdsh28 = SH_C5[3] * xy * (3.0 * z * (52.5 * zz - 7.5) - 30.0 * z)
                    dRGBdsh29 = SH_C5[4] * y * (8 - 28 * (xx+yy)  + 21*( x4 + y4 + 2*xx*yy))
                    dRGBdsh30 = SH_C5[5] * z * (8 - 56 * (xx+ yy) + 63*(x4 + 2 * xx * yy +  y4) )
                    dRGBdsh31 = SH_C5[6] * x * (8 - 28 * (xx + yy) + 21 * (x4 + 2*xx*yy + y4))
                    dRGBdsh32 = SH_C5[7] * z * (yy - xx)* (-2 + 3*xx + 3*yy)
                    dRGBdsh33 = SH_C5[8] * x * (8*zz - xx - yy)*(xx - 3 * yy)
                    dRGBdsh34 = SH_C5[9]  * z * (-6.0 * xx * zz + x4 + y4)
                    dRGBdsh35 = SH_C5[10] * x * (-10.0 * xx * yy + x4 + 5.0 * y4)

                    dL_dsh[...,25] = dRGBdsh25 * dL_dRGB
                    dL_dsh[...,26] = dRGBdsh26 * dL_dRGB
                    dL_dsh[...,27] = dRGBdsh27 * dL_dRGB
                    dL_dsh[...,28] = dRGBdsh28 * dL_dRGB
                    dL_dsh[...,29] = dRGBdsh29 * dL_dRGB
                    dL_dsh[...,30] = dRGBdsh30 * dL_dRGB
                    dL_dsh[...,31] = dRGBdsh31 * dL_dRGB
                    dL_dsh[...,32] = dRGBdsh32 * dL_dRGB
                    dL_dsh[...,33] = dRGBdsh33 * dL_dRGB
                    dL_dsh[...,34] = dRGBdsh34 * dL_dRGB
                    dL_dsh[...,35] = dRGBdsh35 * dL_dRGB

                    # dRGBdx += SH_C5[0] * SphereHarmonics_Coeffs[...,16] * (0.) +\
                    #         # SH_C5[1] * SphereHarmonics_Coeffs[...,17] * (3*xx*y - y3) + \
                    #         # SH_C5[2] * SphereHarmonics_Coeffs[...,18] * (12 * xy*z) + \
                    #         # SH_C5[3] * SphereHarmonics_Coeffs[...,19] * (12*y*zz - 3*xx*y - 3 * y3) + \
                    #         # SH_C5[4] * SphereHarmonics_Coeffs[...,20] * (20*z3-60*z*xx-60*yy*z) + \
                    #         # SH_C5[5] * SphereHarmonics_Coeffs[...,21] * (12*x*zz - 3*x3 - 3 * x*yy) + \
                    #         # SH_C5[6] * SphereHarmonics_Coeffs[...,22] * 12.*z*(xx-yy)+\
                    #         # SH_C5[7] * SphereHarmonics_Coeffs[...,23] * (x3 - 3 * yy *x)+\
                    #         # SH_C5[8] * SphereHarmonics_Coeffs[...,24] * (0.)+\
                    #         # SH_C5[9] * SphereHarmonics_Coeffs[...,24] * (0.)+\
                    #         # SH_C5[10] * SphereHarmonics_Coeffs[...,24] * (0.)
        dL_dposition = dnormvdv3d(dir_orig,
                              direction_v_x= (dRGBdx.squeeze(dim=-1) * dL_dRGB).sum(dim=-1),
                              direction_v_y = (dRGBdy.squeeze(dim=-1) * dL_dRGB).sum(dim=-1),
                              direction_v_z= (dRGBdz.squeeze(dim=-1) * dL_dRGB).sum(dim=-1))

        # print('backward position',dL_dpositions)

        return dL_dposition, None, None, None, dL_dsh,None
    
"""Real spherical harmonics in Cartesian form for PyTorch.

This is an autogenerated file. See
https://github.com/cheind/torch-spherical-harmonics
for more information.
"""
def rsh_cart_0(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 0.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,1) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
        ],
        -1,
    )


def rsh_cart_1(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 1.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,4) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
        ],
        -1,
    )


def rsh_cart_2(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 2.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,9) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
        ],
        -1,
    )


def rsh_cart_3(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 3.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,16) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            -0.590043589926644 * y * (3.0 * x2 - y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (1.5 - 7.5 * z2),
            1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
            0.304697199642977 * x * (1.5 - 7.5 * z2),
            1.44530572132028 * z * (x2 - y2),
            -0.590043589926644 * x * (x2 - 3.0 * y2),
        ],
        -1,
    )


def rsh_cart_4(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 4.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,25) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            -0.590043589926644 * y * (3.0 * x2 - y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (1.5 - 7.5 * z2),
            1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
            0.304697199642977 * x * (1.5 - 7.5 * z2),
            1.44530572132028 * z * (x2 - y2),
            -0.590043589926644 * x * (x2 - 3.0 * y2),
            2.5033429417967 * xy * (x2 - y2),
            -1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.267618617422916 * y * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
            1.48099765681286
            * z
            * (1.66666666666667 * z * (1.5 * z2 - 0.5) - 0.666666666666667 * z)
            - 0.952069922236839 * z2
            + 0.317356640745613,
            0.267618617422916 * x * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            -1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
        ],
        -1,
    )


def rsh_cart_5(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 5.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,36) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z
    x4 = x2**2
    y4 = y2**2
    z4 = z2**2

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            -0.590043589926644 * y * (3.0 * x2 - y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (1.5 - 7.5 * z2),
            1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
            0.304697199642977 * x * (1.5 - 7.5 * z2),
            1.44530572132028 * z * (x2 - y2),
            -0.590043589926644 * x * (x2 - 3.0 * y2),
            2.5033429417967 * xy * (x2 - y2),
            -1.77013076977993 * yz * (3.0 * x2 - y2),
            0.126156626101008 * xy * (52.5 * z2 - 7.5),
            0.267618617422916 * y * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
            1.48099765681286
            * z
            * (1.66666666666667 * z * (1.5 * z2 - 0.5) - 0.666666666666667 * z)
            - 0.952069922236839 * z2
            + 0.317356640745613,
            0.267618617422916 * x * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z),
            0.063078313050504 * (x2 - y2) * (52.5 * z2 - 7.5),
            -1.77013076977993 * xz * (x2 - 3.0 * y2),
            -3.75501441269506 * x2 * y2
            + 0.625835735449176 * x4
            + 0.625835735449176 * y4,
            -0.65638205684017 * y * (-10.0 * x2 * y2 + 5.0 * x4 + y4),
            8.30264925952416 * xy * z * (x2 - y2),
            0.00931882475114763 * y * (52.5 - 472.5 * z2) * (3.0 * x2 - y2),
            0.0913054625709205 * xy * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z),
            0.241571547304372
            * y
            * (
                2.25 * z * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                + 9.375 * z2
                - 1.875
            ),
            -1.24747010616985 * z * (1.5 * z2 - 0.5)
            + 1.6840846433293
            * z
            * (
                1.75
                * z
                * (1.66666666666667 * z * (1.5 * z2 - 0.5) - 0.666666666666667 * z)
                - 1.125 * z2
                + 0.375
            )
            + 0.498988042467941 * z,
            0.241571547304372
            * x
            * (
                2.25 * z * (2.33333333333333 * z * (1.5 - 7.5 * z2) + 4.0 * z)
                + 9.375 * z2
                - 1.875
            ),
            0.0456527312854602 * (x2 - y2) * (3.0 * z * (52.5 * z2 - 7.5) - 30.0 * z),
            0.00931882475114763 * x * (52.5 - 472.5 * z2) * (x2 - 3.0 * y2),
            2.07566231488104 * z * (-6.0 * x2 * y2 + x4 + y4),
            -0.65638205684017 * x * (-10.0 * x2 * y2 + x4 + 5.0 * y4),
        ],
        -1,
    )
if __name__ == "__main__":
    from torch.autograd import gradcheck
    position = torch.randn(1,5,3,requires_grad=True).double().cuda()
    degree = 4
    max_coeffs = 25
    camera_positions = torch.tensor([0.,0.,-0.]).double().cuda()
    SphereHarmonics_Coeffs = torch.randn(1,5,6,max_coeffs,requires_grad=True).double().cuda()

    test = gradcheck(SphereHarmonic.apply,(position,degree,max_coeffs,camera_positions,SphereHarmonics_Coeffs),
                     eps=1e-6,atol=1e-6,raise_exception=True)
    print('111',test)
