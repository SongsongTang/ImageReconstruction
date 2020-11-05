import numpy as np
import scipy.io as sio

def rl_filter(n, T):
    """
        R-L滤波器
        n为采样点数
        T为采样间隔
    """
    f = np.zeros(np.shape(n))
    j = 0
    for i in n:
        i = int(i)
        if i == 0:
            f[j] = 1/(2*T)**2
        elif i%2:
            f[j] = -1/(i*np.pi*T)**2
        else:
            f[j] = 0
        j += 1
    return f

def convolve2D(two_dimentions, one_dimention):
    """
        计算二维投影数据和滤波器卷积
    """
    convolved_data = np.zeros(np.shape(two_dimentions))
    for i in range(np.shape(two_dimentions)[0]):
        # 卷积返回数组尺寸为max(M, N) - min(M, N) + 1
        convolved_data[i] = np.convolve(two_dimentions[i], one_dimention, mode="valid")
    convolved_data
    return convolved_data

def main():
    # 加载投影真值
    true_contents = sio.loadmat("data/proj_Shepplogan_360_300.mat")["proj_Shepplogan_360_300"]
    # 加载大作业（1）投影值
    proj_contents = sio.loadmat("data/siddon_projected_Shepplogan.mat")["proj_Shepplogan"]
    # 每个像素大小
    pixel = 20/256
    # 获取x， y， β数组
    x_array = np.linspace(-10, 10 - pixel, 256)
    y_array = np.linspace(-10 + pixel, 10, 256)[:, np.newaxis]
    beta_array = np.arange(0, 2 * np.pi, np.pi / 180)
    # 设置R为60
    R = 60
    # 设置采样点数
    n = np.linspace(-299, 299, 599)
    # 采样间隔
    T = 0.08
    # 定义滤波器
    h=rl_filter(n, T)/2
    # 定义探测器序列
    ai = np.linspace(-149.5, 149.5, 300)
    a_array = ai * 0.08
    # 对投影值进行加权
    pwd = true_contents * R / (R**2+a_array **2)**(1/2)
    pwd = proj_contents * R / (R**2+a_array **2)**(1/2)
    # 将投影值与滤波器卷积
    conv = convolve2D(pwd, h)
    # 初始化头模型数组
    fbp = np.zeros((256,256))
    # 对每一个旋转角度做滤波反投影
    for beta in beta_array:
        # 计算U
        U = R + x_array*np.cos(beta) + y_array*np.sin(beta)
        # 计算a
        a = R/U*(-x_array*np.sin(beta)+y_array*np.cos(beta))
        # 将不在探测器位置的a值向下取到探测器位置，并将y以x轴镜像
        a_fit = np.array((((a-0.04) // T) * T + 0.04)[::-1, :])
        # 将探测器位置之外的a值取到探测器上
        a_fit[a_fit > 11.96] = 11.96
        a_fit[a_fit < -11.96] = -11.96
        # 获取a的序列
        ai = ((a_fit + 11.96) / T).astype(int)
        # 计算一个旋转角度下的滤波反投影值
        fi = R**2/U**2*conv[int(beta*180/np.pi), ai]
        # 将其累加到头模型数组
        fbp += fi
    # 保存.mat文件
    mdic = {'__version__': '1.0', '__globals__': [], 'Shepplogan': fbp}
    sio.savemat("./data/FBP_Shepplogan.mat", mdic)

if __name__ == '__main__':
    main()
