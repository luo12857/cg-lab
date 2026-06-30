import taichi as ti
import time  # 新增：用于打印时间

# 立即打印启动信息（解决“没结果”的直观问题）
print("🔍 程序已启动，正在初始化 Taichi...")
print(f"⏰ 当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 注意：初始化必须在最前面执行，接管底层 GPU
ti.init(arch=ti.gpu)
print("✅ Taichi 初始化完成（GPU 模式）")

# 导入我们自己写的模块
from src.Work0.config import WINDOW_RES, PARTICLE_COLOR, PARTICLE_RADIUS
from src.Work0.physics import init_particles, update_particles, pos

def run():
    print("⚙️ 正在编译 GPU 内核，请稍候...")
    init_particles()
    
    gui = ti.GUI("Experiment 0: Taichi Gravity Swarm", res=WINDOW_RES)
    print("🎉 编译完成！GUI 窗口已弹出，请在窗口中移动鼠标。")
    print("💡 提示：关闭 GUI 窗口即可退出程序")
    
    # 渲染主循环
    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        
        # 驱动 GPU 进行物理计算
        update_particles(mouse_x, mouse_y)
        
        # 读取显存数据并绘制
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()

if __name__ == "__main__":
    run()