import scipy.io as sio
import numpy as np


def get_cross_coordinates(alpha, beta):
    """
    获取射线与模型的交叉坐标
    """
    pixel = 20/256
    # 确定射线参数
    theta = alpha + beta + np.pi/2
    t = 60 * np.sin(alpha)
    # 以x轴定点
    x_x = np.arange(-10, 10+pixel, pixel)
    y_x = (t - x_x*np.cos(theta))/np.sin(theta)
    # 设置y轴范围
    y_lower_limit = y_x >= -10
    y_upper_limit = y_x <= 10
    y_limit = y_upper_limit & y_lower_limit
    # 取交点坐标
    x_x_c = x_x[y_limit]
    y_x_c = y_x[y_limit]
    # 以y轴定点
    y_y = np.arange(-10, 10+pixel, pixel)
    x_y = (t - y_y*np.sin(theta))/np.cos(theta)
    # 设置x轴范围
    x_lower_limit = x_y >= -10
    x_upper_limit = x_y <= 10
    x_limit = x_upper_limit & x_lower_limit
    # 取交点坐标
    x_y_c = x_y[x_limit]
    y_y_c = y_y[x_limit]
    # 将x、y轴交点合并，并转置，按x值排序
    x_c = np.append(x_x_c, x_y_c)
    y_c = np.append(y_x_c, y_y_c)
    cross_point = np.vstack((x_c, y_c))
    cross_point_unique = np.unique(cross_point, axis=1)
    cp = cross_point_unique.transpose()
    cp_sorted = cp[cp[:,0].argsort()]
    return cp_sorted

def get_projected_values(cp_sorted, mat_contents):
    """
    计算相邻两点距离，两点所占方格的值
    """
    point_num = np.shape(cp_sorted)[0]
    distance = []
    values = []
    for i in range(point_num-1):
        # 取出两相邻点
        x1 = cp_sorted[i, 0]
        x2 = cp_sorted[i+1, 0]
        y1 = cp_sorted[i, 1]
        y2 = cp_sorted[i+1, 1]
        # 计算两相邻点距离
        dis = ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)
        distance.append(dis)
        # 确定两相邻点所在方格
        n1 = int((x1+10)/(5/64))
        m1 = int((y1-10)/(-5/64))
        n2 = int((x2+10)/(5/64))
        m2 = int((y2-10)/(-5/64))
        m = min(m1, m2)
        n = min(n1, n2)
        # 获取方格的值
        value = mat_contents[m, n]
        values.append(value)
    ndistance = np.array(distance)
    nvalues = np.array(values)
    # 根据siddon方法计算投影值
    projected_value = np.sum(distance * nvalues)
    return projected_value

def main():
    # 加载Shepp_logan模型
    mat_fname = 'Shepp_logan.mat'
    mat_contents = sio.loadmat(mat_fname)['Shepp_logan']
    # 每束射线都打到探测器中心，则所有探测器序列
    y = np.arange(-17.94, 17.94, 0.12)
    # 射线与探测器阵列夹角的余角序列
    alpha_list = np.arctan(y/90)
    # X光机靶点和探测器阵列绕头模型中心旋转的角度序列
    beta_list = np.arange(-2*np.pi, 0, np.pi/180)
    # 定义空的投影数据容器，用于存放后续数据
    projected_data = []
    # 初始化程序运行进度
    n = 0
    # 对每一个旋转角度进行投影
    for beta in beta_list:
        data_by_beta = []
        # 对于每一个探测器进行投影
        for alpha in alpha_list:
            # 获取射线与头模型边界的交点坐标
            coordinates = get_cross_coordinates(alpha, beta)
            # 计算投影值
            values = get_projected_values(coordinates, mat_contents)
            data_by_beta.append(values)
            # 显示进度
            print("Progress: " + str(round(n/1080, 2)) + "%", end="\r")
            n += 1
        # 将投影数据保存到容器内
        projected_data.append(data_by_beta)
    # 输出为.mat文件
    mdic = {'__version__': '1.0', '__globals__': [], 'proj_Shepplogan': projected_data}
    sio.savemat("siddon_projected_Shepplogan.mat", mdic)

if __name__ == "__main__":
    main()
