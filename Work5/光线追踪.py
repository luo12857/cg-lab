import taichi as ti

# 初始化 Taichi GPU 后端 (Mac 自动调用 Metal，Win 调用 CUDA/Vulkan)
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# MSAA 采样数
MSAA_SAMPLES = 2

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

# 玻璃材质折射率
GLASS_IOR = 1.5

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract(I, N, eta):
    """
    斯涅尔定律计算折射方向
    I: 入射方向 (指向表面)
    N: 法线 (朝外)
    eta: 相对折射率 (n1/n2)
    返回: (折射方向, 是否成功折射)
    """
    cosi = ti.math.clamp(I.dot(N), -1.0, 1.0)
    sin2t = eta * eta * (1.0 - cosi * cosi)
    cost = ti.sqrt(ti.max(0.0, 1.0 - sin2t))
    # 全反射条件
    is_total_reflection = sin2t > 1.0
    # 计算折射方向 (无论是否全反射都计算)
    refract_dir = eta * I + (eta * cosi - cost) * N
    return refract_dir, not is_total_reflection

@ti.func
def fresnel(I, N, ior):
    """
    菲涅尔近似 (Schlick's approximation)
    返回反射概率
    """
    cosi = ti.math.clamp(-1.0, 1.0, I.dot(N))
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - ti.abs(cosi), 5.0)

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0]) # 法线永远朝上
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (t, 法线 N, 颜色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测红色玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.95, 1.0])  # 略带蓝色的玻璃
        hit_mat = MAT_GLASS

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9]) # 镜面反射基础色
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        # 生成棋盘格纹理
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        # 判断坐标的奇偶性来交替颜色
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3]) # 灰色格子
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8]) # 白色格子

    return min_t, hit_n, hit_c, hit_mat

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0])
        
        # MSAA: 对每个像素进行多次采样
        for sample_idx in range(MSAA_SAMPLES):
            # 像素内随机偏移 (-0.5 到 0.5)
            offset_x = ti.random() - 0.5
            offset_y = ti.random() - 0.5
            
            u = (i + offset_x - res_x / 2.0) / res_y * 2.0
            v = (j + offset_y - res_y / 2.0) / res_y * 2.0
            
            ro = ti.Vector([0.0, 1.0, 5.0])  # 摄像机稍微抬高一点
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))  # 视角微微向下看

            sample_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])  # 光线能量吞吐量
            
            # 迭代式光线追踪（代替递归）
            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)
                
                # 如果没击中任何物体，加上背景色并结束追踪
                if t > 1e9:
                    sample_color += throughput * bg_color
                    break
                    
                p = ro + rd * t
                
                # 分支 1：镜面反射材质
                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color
                    
                # 分支 2：玻璃材质（折射）
                elif mat_id == MAT_GLASS:
                    cosi = ti.math.clamp(rd.dot(N), -1.0, 1.0)
                    # 根据入射方向计算折射率和反射法线
                    eta = (1.0 / GLASS_IOR) if cosi < 0 else GLASS_IOR
                    N_refl = N if cosi < 0 else -N
                    
                    # 菲涅尔反射率
                    F = fresnel(rd, N_refl, GLASS_IOR)
                    
                    # 随机决定反射还是折射（按菲涅尔权重）
                    if ti.random() < F:
                        # 反射
                        ro = p + N * 1e-4
                        rd = normalize(reflect(rd, N_refl))
                    else:
                        # 折射
                        refract_dir, success = refract(rd, N_refl, eta)
                        if success:
                            ro = p - N_refl * 1e-4
                            rd = normalize(refract_dir)
                        else:
                            # 全反射
                            ro = p + N * 1e-4
                            rd = normalize(reflect(rd, N_refl))
                    
                    throughput *= 0.95 * obj_color  # 玻璃吸收少量能量
                    
                # 分支 3：漫反射材质
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)
                    
                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                    
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0
                        
                    ambient = 0.2 * obj_color
                    direct_light = ambient
                    
                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse
                    
                    sample_color += throughput * direct_light
                    break
            
            final_color += sample_color
        
        # 平均多次采样结果
        pixels[i, j] = ti.math.clamp(final_color / float(MSAA_SAMPLES), 0.0, 1.0)

def main():
    window = ti.ui.Window("Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    # 初始化光源位置和弹射次数
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 3

    while window.running:
        render()
        canvas.set_image(pixels)
        
        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.22):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 5)

        window.show()

if __name__ == '__main__':
    main()