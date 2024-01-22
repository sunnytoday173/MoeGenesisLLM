import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 临时增加字体避免乱码
matplotlib.font_manager.fontManager.addfont('./SimHei.ttf')

# 设置matplotlib的中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或者其它支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

def draw(current_day,current_ability):
    for key in current_ability:
        current_ability[key] = int(current_ability[key])
    # 标签和值
    labels=np.array(list(current_ability.keys()))
    stats=np.array(list(current_ability.values()))
    # 计算每个点的角度
    angles=np.linspace(0+0.5*np.pi, 2*np.pi+0.5*np.pi, len(labels), endpoint=False).tolist()
    angles = [angle%(2*np.pi) for angle in angles]
    print(angles)

    # 使图形闭合
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='red', alpha=0.25)
    ax.plot(angles, stats, color='red', linewidth=2)  # 画线

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])

    # 在外侧显示数值
    for angle, value in zip(angles, stats):
        ax.text(angle, 12, str(value), ha='center', va='center', fontsize=12)

    # 设置雷达图的范围
    ax.set_ylim(0, 10)

    # 添加标题
    plt.title('当前能力值',pad=45)

    # 在显示之前调整布局，增加顶部边距
    plt.subplots_adjust(top=0.85)

    # 显示图形
    #plt.show()
    
    # 保存文件
    plt.savefig(f'./image/{current_day}.jpg')
    
    # 返回存储路径
    return f'./image/{current_day}.jpg'
