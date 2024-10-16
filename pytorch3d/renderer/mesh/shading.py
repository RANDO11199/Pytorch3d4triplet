# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Tuple
import torch.nn.functional as F
import torch
from pytorch3d.ops import interpolate_face_attributes
from ..lighting import SHLights
from .textures import TexturesVertex
import math
from torchvision.utils import save_image,make_grid


def DistributionGGX(N, H, roughness):
    a = roughness*roughness
    a2 = torch.square(a)
    NdotH  = F.relu(torch.sum(N * H, dim=-1,keepdim=True))
    NdotH2 = torch.square(NdotH)
    num = a2
    denom = (NdotH2*(a2-1.0) + 1.0)
    denom = math.pi * denom * denom + 1e-6
    return num / denom

def GeometrySchlickGGX(DotProduct, k):

    num   = DotProduct
    denom = DotProduct * (1.0 - k) + k 
	
    return num / denom

def GeometrySmith(NdotL,NdotV,  roughness):
    r = (roughness + 1.0)
    k = (r*r) / 8.0
    ggx2  = GeometrySchlickGGX(NdotV, k)
    ggx1  = GeometrySchlickGGX(NdotL, k)
	
    return ggx1 * ggx2

def fresnelSchlick( HdotV,  F0): # NdotV or HdotV?

    return F0 + (1.0 - F0) * torch.pow(torch.clamp(1.0 - HdotV, 0.0, 1.0), 5.0)


def _apply_lighting_CookTorrance_Standard(
        points, normals, lights, cameras, materials,fragments,faces):
    
    albedo = materials.diffuse_color
    roughness = materials.shininess
    metallic = materials.specular_color
    ao = materials.ambient_color
    # normals = F.normalize(normals,p=2,dim=-1)
    p2f = fragments.pix_to_face
    p2f = p2f.unsqueeze(dim = -1)
    p2f = p2f[p2f!=-1].reshape(-1) # the code of faces used
    p2f_unique, area_counting = torch.unique(p2f,return_counts = True)
    pixel_face_mask = torch.zeros(faces.shape[0],dtype = torch.bool)
    pixel_face_mask[p2f_unique] = True 
    used_vertexes = faces[pixel_face_mask] # used vertexes,but wrap in faces 
    unique_used_vertexes = used_vertexes.unique()
    faces_verts = points[faces]
    faces_normals = normals[faces]
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    pixel_normals = F.normalize(pixel_normals,p=2,dim=-1)

    radiance,wi = lights.specular(
        points=points,
        camera_position= cameras.get_camera_center(),used_vertex=unique_used_vertexes
    )
    radiances = torch.zeros(points.shape[0],3,device='cuda')
    radiances[unique_used_vertexes] = radiance.reshape(-1,3)
    
    radiance = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, radiances[faces]
    )


    directions = torch.zeros(points.shape[0],3,device='cuda')
    directions[unique_used_vertexes] = wi.reshape(-1,3)

    direction = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, directions[faces]
    )
    direction = F.normalize(direction,p=2,dim=-1)

    view_direction = cameras.get_camera_center() - pixel_coords_in_camera
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-12)

    NdotL = F.relu(torch.sum(pixel_normals * direction, dim=-1,keepdim=True))
    # No specular highlights if angle is less than 0.
    # mask = (NdotL > 0).to(torch.float32)
    
    halfway_vector = direction + view_direction
    halfway_vector = F.normalize(halfway_vector, p=2, dim=-1, eps=1e-12)

    NdotV = F.relu(torch.sum(view_direction * pixel_normals, dim=-1,keepdim=True))
    HdotV = F.relu(torch.sum(view_direction * halfway_vector, dim=-1,keepdim=True))
    NDF = DistributionGGX(N = pixel_normals, H = halfway_vector,roughness=roughness)

    F0 =0.04*(1-metallic) + metallic * (albedo) 

    F_schlick = fresnelSchlick(HdotV=HdotV,F0=F0) #NdotV or HdotV, NdotV save computation
    # F_schlick = fresnelSchlick(HdotV=HdotV,F0=F0) #NdotV or HdotV

    GGX = GeometrySmith(NdotL,NdotV,roughness)
    kd = F.relu(1-F_schlick)  * (1-metallic)

    specular_brdf =  NDF * F_schlick * GGX /(4*(NdotV)*(NdotL) + 1e-4)
    specular_color = specular_brdf * NdotL * radiance 
    diffuse_brdf = ( kd * albedo/math.pi) 
    diffuse_color = diffuse_brdf * NdotL * radiance 
    ambient_color  = lights.ambient_color * albedo * ao  # AO

    return ambient_color,diffuse_color,specular_color,pixel_normals,[p2f_unique,unique_used_vertexes,area_counting]

def _apply_lighting_CookTorrance_Simple(
        points, normals, lights, cameras, materials,fragments,faces,used_vertexes):
    
    albedo = materials.diffuse_color
    roughness = torch.sigmoid(materials.shininess) 
    ao = materials.ambient_color[...,(0,)]
    # metallic = materials.specular_color

    # normals = F.normalize(normals,p=2,dim=-1)
    p2f = fragments.pix_to_face
    p2f = p2f.unsqueeze(dim = -1)
    p2f = p2f[p2f!=-1].reshape(-1) # the code of faces used
    p2f_unique, area_counting = torch.unique(p2f,return_counts = True)
    pixel_face_mask = torch.zeros(faces.shape[0],dtype = torch.bool)
    pixel_face_mask[p2f_unique] = True 
    used_vertexes = faces[pixel_face_mask] # used vertexes,but wrap in faces 
    unique_used_vertexes = used_vertexes.unique()
    faces_verts = points[faces]
    faces_normals = normals[faces]
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    pixel_normals = F.normalize(pixel_normals,p=2,dim=-1)

    radiance,wi = lights.specular(
        points=points,
        camera_position= cameras.get_camera_center(),used_vertex=unique_used_vertexes
    )
    radiances = torch.zeros(points.shape[0],3,device='cuda')
    radiances[unique_used_vertexes] = radiance.reshape(-1,3)
    
    radiance = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, radiances[faces]
    )


    directions = torch.zeros(points.shape[0],3,device='cuda')
    directions[unique_used_vertexes] = wi.reshape(-1,3)

    direction = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, directions[faces]
    )
    direction = F.normalize(direction,p=2,dim=-1)

    view_direction = cameras.get_camera_center() - pixel_coords_in_camera
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-12)

    NdotL = F.relu(torch.sum(pixel_normals * direction, dim=-1,keepdim=True))
    # No specular highlights if angle is less than 0.
    # mask = (NdotL > 0).to(torch.float32)
    
    halfway_vector = direction + view_direction
    halfway_vector = F.normalize(halfway_vector, p=2, dim=-1, eps=1e-12)

    NdotV = F.relu(torch.sum(view_direction * pixel_normals, dim=-1,keepdim=True))
    HdotV = F.relu(torch.sum(view_direction * halfway_vector, dim=-1,keepdim=True))
    NDF = DistributionGGX(N = pixel_normals, H = halfway_vector,roughness=roughness)

    F0 = materials.specular_color

    F_schlick = fresnelSchlick(HdotV=HdotV,F0=F0) #NdotV or HdotV, NdotV save computation
    # F_schlick = fresnelSchlick(HdotV=HdotV,F0=F0) #NdotV or HdotV

    GGX = GeometrySmith(NdotL,NdotV,roughness)
    # kd = F.relu(1-F_schlick)  * (1-metallic)

    specular_brdf =  NDF * F_schlick * GGX /(4*(NdotV)*(NdotL) + 1e-4)
    specular_color = specular_brdf * NdotL * radiance 
    diffuse_brdf = (albedo/math.pi) 
    diffuse_color = diffuse_brdf * NdotL * radiance 
    ambient_color  = lights.ambient_color * albedo * ao  # AO
    return ambient_color,diffuse_color,specular_color,pixel_normals

def _apply_lighting_BlinnPhong(
        points, normals, lights, cameras, materials,fragments,faces):
    

    p2f = fragments.pix_to_face

    p2f = p2f.unsqueeze(dim = -1)
    p2f = p2f[p2f!=-1].reshape(-1) # the code of faces used
    p2f_unique, area_counting = torch.unique(p2f,return_counts = True)
    pixel_face_mask = torch.zeros(faces.shape[0],dtype = torch.bool)
    pixel_face_mask[p2f_unique] = True 
    used_vertexes = faces[pixel_face_mask] # used vertexes,but wrap in faces 
    unique_used_vertexes = used_vertexes.unique()
    faces_verts = points[faces]
    faces_normals = normals[faces]
    
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    pixel_normals = F.normalize(pixel_normals,p=2,dim=-1, eps=1e-6)

    radiance,wi = lights.specular(
        points=points,
        camera_position= cameras.get_camera_center(),used_vertex=unique_used_vertexes
    )
    radiances = torch.zeros(points.shape[0],3,device='cuda')
    radiances[unique_used_vertexes] = radiance.reshape(-1,3)

    radiance = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, radiances[faces]
    )

    directions = torch.zeros(points.shape[0],3,device='cuda')
    directions[unique_used_vertexes] = wi.reshape(-1,3)

    direction = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, directions[faces]
    )
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)

    view_direction = cameras.get_camera_center() - pixel_coords_in_camera
    view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)

    NdotL = F.relu(torch.sum(pixel_normals * direction, dim=-1,keepdim=True))
    # No specular highlights if angle is less than 0.
    mask = (NdotL > 0).to(torch.float32)
    
    halfway_vector = direction + view_direction
    halfway_vector = F.normalize(halfway_vector, p=2, dim=-1, eps=1e-6)
    shininess = torch.nn.functional.relu(materials.shininess) + 1.
    alpha = F.relu(torch.sum(pixel_normals * halfway_vector, dim=-1,keepdim=True)) * mask

    specular_brdf = materials.specular_color * torch.pow(alpha, shininess )
    diffuse_brdf = materials.diffuse_color * NdotL
    specular_color = radiance * specular_brdf
    diffuse_color = radiance  * diffuse_brdf
    ambient_color  = lights.ambient_color * materials.ambient_color # AO
    return ambient_color,diffuse_color,specular_color,pixel_normals,[p2f_unique,unique_used_vertexes,area_counting]

def _apply_lighting(
    points, normals, lights, cameras, materials
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, ..., 3) or (P, 3).
        normals: torch tensor of shape (N, ..., 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        materials: instance of the Materials class.

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(
        normals=normals,# world coordinate 
        points=points, # world coordinate 
        camera_position=cameras.get_camera_center(), # world coordinate 
        shininess=materials.shininess,
    )
    # ambient_color = materials.ambient_color * lights.ambient_color * lights.ambient_intensity
    # ambient_light_color = 
    normals_dims = normals.shape[1:-1]
    expand_dims = (-1,) + (1,) * len(normals_dims)
    if lights.ambient_intensity.shape != normals.shape:
        ambient_light_intensity = lights.ambient_intensity.view(expand_dims + (3,))
        ambient_light_color = lights.ambient_color.view(expand_dims + (3,))
    # ambient_color = materials.ambient_color * ambient_light_intensity *  ambient_light_color
    ambient_color = materials.ambient_color * ambient_light_color


    diffuse_color = materials.diffuse_color * light_diffuse
    specular_color = materials.specular_color * light_specular

    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(),
            diffuse_color.squeeze(),
            specular_color.squeeze(),
        )

    if ambient_color.ndim != diffuse_color.ndim:
        # Reshape from (N, 3) to have dimensions compatible with
        # diffuse_color which is of shape (N, H, W, K, 3)
        ambient_color = ambient_color[:, None, None, None, :]
    return ambient_color, diffuse_color, specular_color

def _cooktorrance_shading_with_pixels(
    meshes, fragments, lights, cameras, materials, texels =None,**kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
        pixel_coords: (N, H, W, K, 3), camera coordinates of each intersection.
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)

    materials.specular_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.specular_color[:,faces][0]
    )
    materials.ambient_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.ambient_color[:,faces][0]
    )
    materials.diffuse_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.diffuse_color[:,faces][0]
    )    
    materials.shininess = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.shininess[:,faces][0]
    )    

    if isinstance(lights,SHLights):

        ambient, diffuse, specular, pixel_normals,vindex = _apply_lighting_CookTorrance_Standard(verts, vertex_normals, lights, cameras, materials,fragments,faces)
        # colors =  (ambient + diffuse) * texels + specular # Bad feeeling to the texels terms, not happy about it, very contrived. TODO: Figure out why use this term
        # ao_map = texels[...,(0,)]
        colors = (ambient+ diffuse) + specular
        return colors, None,pixel_normals,vindex
    else:
        raise NotImplementedError(f'Light type :{type(lights)} Not implemented for CookTorrance yet')
def _blinnphong_shading_with_pixels(
    meshes, fragments, lights, cameras, materials, texels =None,**kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
        pixel_coords: (N, H, W, K, 3), camera coordinates of each intersection.
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]

    materials.specular_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.specular_color[:,faces][0]
    )
    materials.ambient_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.ambient_color[:,faces][0]
    )
    materials.diffuse_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.diffuse_color[:,faces][0]
    )    
    materials.shininess = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.shininess[:,faces][0]
    )    

    if isinstance(lights,SHLights):

        ambient, diffuse, specular, pixel_normals,vindex = _apply_lighting_BlinnPhong(verts, vertex_normals, lights, cameras, materials,fragments,faces)
        # colors =  (ambient + diffuse) * texels + specular # Bad feeeling to the texels terms, not happy about it, very contrived. TODO: Figure out why use this term
        # ao_map = texels[...,(0,)]
        colors = (ambient+ diffuse) + specular
        return colors, None,pixel_normals,vindex
    else:

        pixel_coords_in_camera = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts
        )
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )
        # 20240824 detect normals nan
        if pixel_normals.isnan().any():
            print('shading, pixel normal ,nan')
        # materials.emission = interpolate_face_attributes(
        #     fragments.pix_to_face, fragments.bary_coords, materials.emission[:,faces][0]
        # )    
        ambient, diffuse, specular = _apply_lighting( # actually phong
            pixel_coords_in_camera, pixel_normals, lights, cameras, materials
        )
        if ambient.isinf().any() or ambient.isnan().any() or  diffuse.isinf().any() or diffuse.isnan().any() or  specular.isinf().any() or specular.isnan().any():
            print('inner')
        colors = (ambient + diffuse) * texels + specular # The specular reflection does not depend on the underlying texture color

        return colors, pixel_coords_in_camera,pixel_normals
    
def _phong_shading_with_pixels(
    meshes, fragments, lights, cameras, materials, texels =None,**kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
        pixel_coords: (N, H, W, K, 3), camera coordinates of each intersection.
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]

    materials.specular_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.specular_color[:,faces][0]
    )
    materials.ambient_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.ambient_color[:,faces][0]
    )
    materials.diffuse_color = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.diffuse_color[:,faces][0]
    )    
    materials.shininess = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, materials.shininess[:,faces][0]
    )    

    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    # 20240824 detect normals nan
    if pixel_normals.isnan().any():
        print('shading, pixel normal ,nan')
    # materials.emission = interpolate_face_attributes(
    #     fragments.pix_to_face, fragments.bary_coords, materials.emission[:,faces][0]
    # )    
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords_in_camera, pixel_normals, lights, cameras, materials
    )
    if ambient.isinf().any() or ambient.isnan().any() or  diffuse.isinf().any() or diffuse.isnan().any() or  specular.isinf().any() or specular.isnan().any():
        print('inner')
    colors = (ambient + diffuse) * texels + specular # The specular reflection does not depend on the underlying texture color

    return colors, pixel_coords_in_camera,pixel_normals


def phong_shading( 
    meshes, fragments, lights, cameras, materials,texels,**kwargs
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    colors, _,pixel_normal,vindex = _phong_shading_with_pixels(
        meshes, fragments, lights, cameras, materials,texels=texels
    )
    return colors,pixel_normal,vindex

def blinnphong_shading( 
    meshes, fragments, lights, cameras, materials,texels,**kwargs
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    colors, _,pixel_normal,vindex = _blinnphong_shading_with_pixels(
        meshes, fragments, lights, cameras, materials,texels=texels
    )
    return colors,pixel_normal,vindex

def cooktorrance_shading( 
    meshes, fragments, lights, cameras, materials,texels,**kwargs
) -> torch.Tensor:
    """
    Apply per pixel shading. First interpolate the vertex normals and
    vertex coordinates using the barycentric coordinates to get the position
    and normal at each pixel. Then compute the illumination for each pixel.
    The pixel color is obtained by multiplying the pixel textures by the ambient
    and diffuse illumination and adding the specular component.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights
        cameras: Cameras class containing a batch of cameras
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    colors, _,pixel_normal,vindex = _cooktorrance_shading_with_pixels(
        meshes, fragments, lights, cameras, materials,texels=texels
    )
    return colors,pixel_normal,vindex

def gouraud_shading(meshes, fragments, lights, cameras, materials) -> torch.Tensor:
    """
    Apply per vertex shading. First compute the vertex illumination by applying
    ambient, diffuse and specular lighting. If vertex color is available,
    combine the ambient and diffuse vertex illumination with the vertex color
    and add the specular component to determine the vertex shaded color.
    Then interpolate the vertex shaded colors using the barycentric coordinates
    to get a color per pixel.

    Gouraud shading is only supported for meshes with texture type `TexturesVertex`.
    This is because the illumination is applied to the vertex colors.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties

    Returns:
        colors: (N, H, W, K, 3)
    """
    if not isinstance(meshes.textures, TexturesVertex):
        raise ValueError("Mesh textures must be an instance of TexturesVertex")

    faces = meshes.faces_packed()  # (F, 3)
    verts = meshes.verts_packed()  # (V, 3)
    verts_normals = meshes.verts_normals_packed()  # (V, 3)
    verts_colors = meshes.textures.verts_features_packed()  # (V, D)
    vert_to_mesh_idx = meshes.verts_packed_to_mesh_idx()

    # Format properties of lights and materials so they are compatible
    # with the packed representation of the vertices. This transforms
    # all tensor properties in the class from shape (N, ...) -> (V, ...) where
    # V is the number of packed vertices. If the number of meshes in the
    # batch is one then this is not necessary.
    if len(meshes) > 1:
        lights = lights.clone().gather_props(vert_to_mesh_idx)
        cameras = cameras.clone().gather_props(vert_to_mesh_idx)
        materials = materials.clone().gather_props(vert_to_mesh_idx)

    # Calculate the illumination at each vertex
    ambient, diffuse, specular = _apply_lighting(
        verts, verts_normals, lights, cameras, materials
    )

    verts_colors_shaded = verts_colors * (ambient + diffuse) + specular
    face_colors = verts_colors_shaded[faces]
    colors = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, face_colors
    )
    return colors


def flat_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
    """
    Apply per face shading. Use the average face position and the face normals
    to compute the ambient, diffuse and specular lighting. Apply the ambient
    and diffuse color to the pixel color and add the specular component to
    determine the final pixel color.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    face_normals = meshes.faces_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    face_coords = faces_verts.mean(dim=-2)  # (F, 3, XYZ) mean xyz across verts

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = fragments.pix_to_face == -1
    pix_to_face = fragments.pix_to_face.clone()
    pix_to_face[mask] = 0

    N, H, W, K = pix_to_face.shape
    idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)

    # gather pixel coords
    pixel_coords = face_coords.gather(0, idx).view(N, H, W, K, 3)
    pixel_coords[mask] = 0.0
    # gather pixel normals
    pixel_normals = face_normals.gather(0, idx).view(N, H, W, K, 3)
    pixel_normals[mask] = 0.0

    # Calculate the illumination at each face
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, cameras, materials
    )
    colors = (ambient + diffuse) * texels + specular
    return colors
