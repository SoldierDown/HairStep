import torch
import numpy as np
import open3d as o3d
import os
import trimesh
from .mesh_util import load_obj_mesh

def hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006):
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor)
        curr_node_orien = curr_node_orien.squeeze()
        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()

def hair_synthesis_DSH(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006, threshold=[60,150]):
    #growing algorithm in DeepSketchHair
    #root:[3, 1024]
    num_strand = root_tensor.shape[2]
    hair_strands = torch.zeros(num_sample, 3, num_strand).to(device=cuda)
    
    curr_node = root_tensor.squeeze()
    last_node_orien = 0
    hair_strands[0] = curr_node
    for i in range(1,num_sample-1):
        curr_node_orien = net.query(curr_node.unsqueeze(0), calib_tensor).squeeze()

        if i>1:
            len_cd = torch.norm(curr_node_orien,p=2,dim=0)
            len_pd = torch.norm(last_node_orien,p=2,dim=0)
            in_prod = torch.sum(curr_node_orien * last_node_orien, dim=0)
            theta = torch.acos( in_prod/ (len_cd*len_pd))*180/np.pi

            idx_big_theta = theta > threshold[1]
            idx_mid_theta = ((theta > threshold[0]).float() - idx_big_theta.float()).bool().unsqueeze(0)#60<theta<150
            idx_stop = (idx_big_theta + torch.isnan(theta).float()).bool().unsqueeze(0)#orien=0 or theta>150

            idx_stop = torch.cat((idx_stop,idx_stop,idx_stop),dim=0)
            idx_mid_theta = torch.cat((idx_mid_theta,idx_mid_theta,idx_mid_theta),dim=0)
            

            half_node_orien = (curr_node_orien + last_node_orien)/2

            orien_zeros = torch.zeros_like(curr_node_orien).float().to(device=cuda)
            curr_node_orien = torch.where(idx_stop, orien_zeros, curr_node_orien)
            curr_node_orien = torch.where(idx_mid_theta, half_node_orien, curr_node_orien)

        hair_strands[i] = hair_strands[i-1] + hair_unit * curr_node_orien
        curr_node = hair_strands[i]
        last_node_orien = curr_node_orien

    return hair_strands.permute(2, 0, 1).cpu().detach().numpy()


def save_strands_with_mesh(strands, mesh_path, outputpath, err=0.3, is_eval=False):
    mesh = trimesh.load(mesh_path, process=False)
    #for coarse mesh /1000.0
    if is_eval:
        mesh.vertices = mesh.vertices/1000.0

    lst_pc_all_valid = []
    lst_num_pt = []
    pc_all_valid = []
    lines = []
    sline = 0

    for i in range(strands.shape[0]):
        current_pc_all_valid = []
        first_step = strands[i,0] - strands[i,1]
        first_step = np.dot(first_step, first_step)
        if np.dot(strands[i,0], strands[i,0])<0.001 or np.dot(strands[i,1], strands[i,1])<0.001:
            continue
        num_pt = 2
        current_pc_all_valid.append(strands[i][0])
        current_pc_all_valid.append(strands[i][1])

        pts_in_out = mesh.contains(strands[i])

        for j in range(2,strands.shape[1]):
            if pts_in_out[j]:
                current_pc_all_valid.append(strands[i][j])
                num_pt += 1
            else:
                break
        lst_pc_all_valid.append(current_pc_all_valid)
        lst_num_pt.append(num_pt)

    min_num_pts = int(sum(lst_num_pt)/len(lst_num_pt)*err)

    for i in range(strands.shape[0]):
        if lst_num_pt[i]<min_num_pts:
            continue
        pc_all_valid = pc_all_valid + lst_pc_all_valid[i]

        for j in range(lst_num_pt[i]-1):
            lines.append([sline + j, sline + j + 1])
        sline += lst_num_pt[i]

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pc_all_valid)), lines=o3d.utility.Vector2iVector(lines))
    o3d.io.write_line_set(outputpath, line_set)

def get_sdf_value(sdf, x):
    x0 = int(x[0])
    y0 = int(x[1])
    z0 = int(x[2])
    
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    dx = x[0] - x0
    dy = x[1] - y0
    dz = x[2] - z0

    c000 = sdf[x0][y0][z0]
    c001 = sdf[x0][y0][z1]
    c010 = sdf[x0][y1][z0]
    c011 = sdf[x0][y1][z1]
    c100 = sdf[x1][y0][z0]
    c101 = sdf[x1][y0][z1]
    c110 = sdf[x1][y1][z0]
    c111 = sdf[x1][y1][z1]

    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy

    c = c0 * (1 - dz) + c1 * dz

    return c

def save_strands_with_sdf(strands, sdf, outputpath, render_path, err=0.3, is_eval=False):
    #for coarse mesh /1000.0
    # if is_eval:
    #     mesh.vertices = mesh.vertices/1000.0

    lst_pc_all_valid = []
    lst_num_pt = []
    pc_all_valid = []
    lines = []
    sline = 0

    resX, resY, resZ = sdf.shape[0], sdf.shape[1], sdf.shape[2]
    b_min = np.array([-0.3, 1.0, -0.3])
    b_max = np.array([0.3, 2.0, 0.3])
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    
    length = b_max - b_min
    
    coords_matrix[0, 0] = resX / length[0]
    coords_matrix[1, 1] = resY / length[1]
    coords_matrix[2, 2] = resZ / length[2]
    coords_matrix[0:3, 3] = b_min

    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    mins = 100.
    maxs = -100.
    for i in range(strands.shape[0]):
        current_pc_all_valid = []
        first_step = strands[i,0] - strands[i,1]
        first_step = np.dot(first_step, first_step)
        if np.dot(strands[i,0], strands[i,0])<0.001 or np.dot(strands[i,1], strands[i,1])<0.001:
            continue
        num_pt = 2
        current_pc_all_valid.append(strands[i][0])
        current_pc_all_valid.append(strands[i][1])

        for j in range(2,strands.shape[1]):
            cur_pos = strands[i][j]
            cur_loc = np.matmul(coords_matrix[:3, :3], cur_pos - b_min)
            cur_sdf = get_sdf_value(sdf, cur_loc)
            if cur_sdf > maxs:
                maxs = cur_sdf
            if cur_sdf < mins:
                mins = cur_sdf
            if cur_sdf == 1.:
                current_pc_all_valid.append(strands[i][j])
                num_pt += 1
            else:
                break
        lst_pc_all_valid.append(current_pc_all_valid)
        lst_num_pt.append(num_pt)
    print('\nsdf: {} to {}'.format(mins, maxs))
    min_num_pts = int(sum(lst_num_pt)/len(lst_num_pt)*err)

    for i in range(strands.shape[0]):
        if lst_num_pt[i]<min_num_pts:
            continue
        pc_all_valid = pc_all_valid + lst_pc_all_valid[i]

        for j in range(lst_num_pt[i]-1):
            lines.append([sline + j, sline + j + 1])
        sline += lst_num_pt[i]
    
    with open(render_path, "w") as file:
        file.write('{}\n'.format(strands.shape[0]))
        for i in range(strands.shape[0]):
            if lst_num_pt[i]<min_num_pts:
                continue
            file.write('{}\n'.format(lst_num_pt[i]))
            for j in range(lst_num_pt[i]):
                file.write('{} {} {}\n'.format(lst_pc_all_valid[i][j][0], lst_pc_all_valid[i][j][1], lst_pc_all_valid[i][j][2]))


    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pc_all_valid)), lines=o3d.utility.Vector2iVector(lines))
    o3d.io.write_line_set(outputpath, line_set)


def save_strands(strands, outputpath, err=0.3, is_eval=False):
    lst_pc_all_valid = []
    lst_num_pt = []
    pc_all_valid = []
    lines = []
    sline = 0

    for i in range(strands.shape[0]):
        current_pc_all_valid = []
        first_step = strands[i,0] - strands[i,1]
        first_step = np.dot(first_step, first_step)
        if np.dot(strands[i,0], strands[i,0])<0.001 or np.dot(strands[i,1], strands[i,1])<0.001:
            continue
        num_pt = 2
        current_pc_all_valid.append(strands[i][0])
        current_pc_all_valid.append(strands[i][1])

        for j in range(2,strands.shape[1]):
            current_pc_all_valid.append(strands[i][j])
            num_pt += 1
        lst_pc_all_valid.append(current_pc_all_valid)
        lst_num_pt.append(num_pt)

    min_num_pts = int(sum(lst_num_pt)/len(lst_num_pt)*err)

    for i in range(strands.shape[0]):
        if lst_num_pt[i]<min_num_pts:
            continue
        pc_all_valid = pc_all_valid + lst_pc_all_valid[i]

        for j in range(lst_num_pt[i]-1):
            lines.append([sline + j, sline + j + 1])
        sline += lst_num_pt[i]

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.asarray(pc_all_valid)), lines=o3d.utility.Vector2iVector(lines))
    o3d.io.write_line_set(outputpath, line_set)

def get_hair_root(filepath='./data/roots10k.obj'):
    root, _ = load_obj_mesh(filepath)
    return root.T

def export_hair_real(net, cuda, data, mesh_path, save_path):
    image_tensor = data['hairstep'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    root_tensor = torch.from_numpy(get_hair_root()).to(device=cuda).float().unsqueeze(0)

    net.filter(image_tensor)

    strands = hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006)
    save_strands_with_mesh(strands, mesh_path, save_path, 0.3)

def export_hair_real_no_mesh(net, cuda, data, save_path):
    image_tensor = data['hairstep'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    root_tensor = torch.from_numpy(get_hair_root()).to(device=cuda).float().unsqueeze(0)

    net.filter(image_tensor)

    strands = hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006)
    save_strands(strands, save_path, 0.3)

def export_hair_real_with_sdf(net, cuda, data, sdf, save_path, render_path):
    image_tensor = data['hairstep'].to(device=cuda).unsqueeze(0)
    calib_tensor = data['calib'].to(device=cuda).unsqueeze(0)
    root_tensor = torch.from_numpy(get_hair_root()).to(device=cuda).float().unsqueeze(0)

    net.filter(image_tensor)

    strands = hair_synthesis(net, cuda, root_tensor, calib_tensor, num_sample=100, hair_unit=0.006)
    save_strands_with_sdf(strands, sdf, save_path, render_path, 0.3)
