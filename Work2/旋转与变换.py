import taichi as ti
import math

ti.init(arch=ti.gpu)  # 优先使用GPU加速

# 窗口分辨率
RESOLUTION = (700, 700)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=RESOLUTION)

# -------------------------- 全局场定义（关键修复：移到kernel外） --------------------------
# 立方体顶点场（提前定义，避免在kernel内创建）
cube_vertices = ti.Vector.field(4, dtype=ti.f32, shape=8)
# 立方体边场（提前定义）
cube_edges = ti.Vector.field(2, dtype=ti.i32, shape=12)

# -------------------------- 初始化立方体数据（独立函数） --------------------------
@ti.kernel
def init_cube_data():
    """初始化立方体顶点和边数据（仅运行一次）"""
    # 立方体顶点（8个，中心在原点，边长2）
    cube_vertices[0] = ti.Vector([-1, -1, -1, 1])
    cube_vertices[1] = ti.Vector([ 1, -1, -1, 1])
    cube_vertices[2] = ti.Vector([ 1,  1, -1, 1])
    cube_vertices[3] = ti.Vector([-1,  1, -1, 1])
    cube_vertices[4] = ti.Vector([-1, -1,  1, 1])
    cube_vertices[5] = ti.Vector([ 1, -1,  1, 1])
    cube_vertices[6] = ti.Vector([ 1,  1,  1, 1])
    cube_vertices[7] = ti.Vector([-1,  1,  1, 1])
    
    # 立方体边（12条）
    cube_edges[0] = ti.Vector([0, 1])
    cube_edges[1] = ti.Vector([1, 2])
    cube_edges[2] = ti.Vector([2, 3])
    cube_edges[3] = ti.Vector([3, 0])
    cube_edges[4] = ti.Vector([4, 5])
    cube_edges[5] = ti.Vector([5, 6])
    cube_edges[6] = ti.Vector([6, 7])
    cube_edges[7] = ti.Vector([7, 4])
    cube_edges[8] = ti.Vector([0, 4])
    cube_edges[9] = ti.Vector([1, 5])
    cube_edges[10] = ti.Vector([2, 6])
    cube_edges[11] = ti.Vector([3, 7])

# -------------------------- 核心矩阵函数（实验要求补全） --------------------------
@ti.func
def get_model_matrix(angle):
    """绕Z轴旋转的模型变换矩阵（角度制转弧度制）"""
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    # 4x4齐次变换矩阵（绕Z轴旋转）
    model = ti.Matrix([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])
    return model

@ti.func
def get_view_matrix(eye_pos):
    """视图变换矩阵：将相机平移至原点（右乘规则）"""
    view = ti.Matrix([
        [1, 0, 0, -eye_pos[0]],
        [0, 1, 0, -eye_pos[1]],
        [0, 0, 1, -eye_pos[2]],
        [0, 0, 0, 1]
    ])
    return view

@ti.func
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    """透视投影矩阵：透视转正交 + 正交投影"""
    # 1. 角度转弧度
    fov_rad = eye_fov * math.pi / 180.0
    # 2. 计算视锥体边界
    n = -zNear
    f = -zFar
    t = ti.tan(fov_rad / 2) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    # 3. 透视转正交矩阵
    persp2ortho = ti.Matrix([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, n+f, -n*f],
        [0, 0, 1, 0]
    ])
    
    # 4. 正交投影矩阵
    ortho_scale = ti.Matrix([
        [2/(r-l), 0, 0, 0],
        [0, 2/(t-b), 0, 0],
        [0, 0, 2/(n-f), 0],
        [0, 0, 0, 1]
    ])
    ortho_trans = ti.Matrix([
        [1, 0, 0, -(r+l)/2],
        [0, 1, 0, -(t+b)/2],
        [0, 0, 1, -(n+f)/2],
        [0, 0, 0, 1]
    ])
    ortho = ortho_scale @ ortho_trans
    
    # 5. 最终投影矩阵
    projection = ortho @ persp2ortho
    return projection

# -------------------------- 辅助函数 --------------------------
@ti.func
def draw_line(p1, p2, color):
    """绘制线段（Bresenham算法）"""
    x0, y0 = int(p1[0]), int(p1[1])
    x1, y1 = int(p2[0]), int(p2[1])
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < RESOLUTION[0] and 0 <= y0 < RESOLUTION[1]:
            pixels[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

@ti.func
def ndc2screen(v):
    """NDC坐标转屏幕坐标"""
    return ti.Vector([
        (v.x + 1.0) * RESOLUTION[0] / 2.0,
        (v.y + 1.0) * RESOLUTION[1] / 2.0
    ])

# -------------------------- 渲染函数（最终版） --------------------------
@ti.kernel
def render(angle: ti.f32, render_cube: ti.i32):
    """渲染函数：MVP变换 + 绘制三角形/立方体"""
    # 清空画布
    pixels.fill(0.0)
    
    # 1. 初始化MVP矩阵
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    projection = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = projection @ view @ model
    
    # -------------------------- 绘制3D三角形 --------------------------
    if render_cube == 0:
        # 三角形顶点
        v0 = ti.Vector([2.0, 0.0, -2.0, 1.0])
        v1 = ti.Vector([0.0, 2.0, -2.0, 1.0])
        v2 = ti.Vector([-2.0, 0.0, -2.0, 1.0])
        
        # MVP变换 + 透视除法
        v0_ndc = mvp @ v0
        v1_ndc = mvp @ v1
        v2_ndc = mvp @ v2
        v0_ndc /= v0_ndc.w
        v1_ndc /= v1_ndc.w
        v2_ndc /= v2_ndc.w
        
        # 转屏幕坐标并绘制
        p0 = ndc2screen(v0_ndc)
        p1 = ndc2screen(v1_ndc)
        p2 = ndc2screen(v2_ndc)
        draw_line(p0, p1, ti.Vector([1.0, 0.0, 0.0]))
        draw_line(p1, p2, ti.Vector([0.0, 1.0, 0.0]))
        draw_line(p2, p0, ti.Vector([0.0, 0.0, 1.0]))
    
    # -------------------------- 绘制3D立方体 --------------------------
    else:
        # 遍历立方体边（使用全局预定义的场）
        for i in range(12):
            v_a = cube_vertices[cube_edges[i][0]]
            v_b = cube_vertices[cube_edges[i][1]]
            
            # MVP变换 + 透视除法
            a_ndc = mvp @ v_a
            b_ndc = mvp @ v_b
            a_ndc /= a_ndc.w
            b_ndc /= b_ndc.w
            
            # 转屏幕坐标并绘制
            p_a = ndc2screen(a_ndc)
            p_b = ndc2screen(b_ndc)
            draw_line(p_a, p_b, ti.Vector([1.0, 1.0, 1.0]))

# -------------------------- 主程序 --------------------------
def main():
    # 初始化立方体数据（仅执行一次）
    init_cube_data()
    
    gui = ti.GUI("MVP Transform (Triangle/Cube)", res=RESOLUTION)
    angle = 0.0
    render_cube = 0
    
    while gui.running:
        # 交互控制
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break
            elif gui.event.key == 'c':
                render_cube = 1 - render_cube
        
        # 旋转控制
        if gui.is_pressed('a'):
            angle += 1.0
        if gui.is_pressed('d'):
            angle -= 1.0
        
        # 渲染
        render(angle, render_cube)
        
        # 显示画面
        gui.set_image(pixels)
        gui.show()

if __name__ == "__main__":
    main()