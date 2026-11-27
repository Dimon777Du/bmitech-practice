import json
import matplotlib.pyplot as plt
import numpy as np

# Загружаем метрики
with open('workspace/works/nut-01/out/metrics.json', 'r', encoding='utf-8') as f:
    deepseek = json.load(f)
with open('workspace/works/nut-01/out/metrics_bigpickle.json', 'r', encoding='utf-8') as f:
    bigpickle = json.load(f)

# Собираем ошибки по калориям для каждого блюда
deepseek_errors = [d['errors_percent']['kcal'] for d in deepseek['dishes']]
bigpickle_errors = [d['errors_percent']['kcal'] for d in bigpickle['dishes'] if d['status'] == 'ok']

# Строим boxplot
fig, ax = plt.subplots(figsize=(10, 6))
data = [deepseek_errors, bigpickle_errors]
labels = ['DeepSeek V4.1 Flash Free', 'Big Pickle']
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)

# Красим боксы
colors = ['#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Добавляем точки (выбросы)
for i, d in enumerate(data):
    x = np.random.normal(i + 1, 0.04, size=len(d))
    ax.scatter(x, d, alpha=0.5, color='black', s=15)

ax.set_ylabel('Ошибка по калориям (MAE %)')
ax.set_title('Сравнение ошибок по калориям: DeepSeek vs Big Pickle')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.axhline(y=15, color='green', linestyle='--', label='Порог 15%')
ax.legend()

# Сохраняем
plt.tight_layout()
plt.savefig('workspace/works/nut-01/report/boxplot.png', dpi=150)
print('Boxplot сохранен: workspace/works/nut-01/report/boxplot.png')
