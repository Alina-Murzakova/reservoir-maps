import os
import sys
import psutil
import re
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_available_memory(mode='auto', custom_limit_gb=None):
    """
    Determines available memory based on environment and mode

    Args:
        mode: 'safe', 'auto', 'aggressive' - memory usage mode
        custom_limit_gb: user-defined limit (overrides auto-detection)

    Returns:
        dict: information about available memory
"""
    result = {
        'environment': 'unknown',
        'total_ram_gb': None,
        'available_ram_gb': None,
        'jupyter_buffer_gb': None,
        'effective_limit_gb': None,
        'limit_source': None,
        'mode': mode,
        'safety_factors': {'safe': 0.3, 'auto': 0.5, 'aggressive': 0.7}
    }

    # 1. Определяем среду
    if 'ipykernel' in sys.modules:
        result['environment'] = 'jupyter'
    elif getattr(sys, 'frozen', False):
        result['environment'] = 'exe'
    else:
        result['environment'] = 'console'

    # 2. Получаем информацию о RAM
    mem = psutil.virtual_memory()
    result['total_ram_gb'] = mem.total / 1024 ** 3
    result['available_ram_gb'] = mem.available / 1024 ** 3

    # 3. Для Jupyter - получаем buffer_size
    if result['environment'] == 'jupyter':
        result['jupyter_buffer_gb'] = get_jupyter_buffer_size_gb()

        # В Jupyter главное ограничение - buffer_size
        if result['jupyter_buffer_gb']:
            result['limit_source'] = 'jupyter_buffer'
            result['effective_limit_gb'] = result['jupyter_buffer_gb']
        else:
            # Если не нашли buffer, используем RAM с большим запасом
            result['limit_source'] = 'ram_conservative'
            result['effective_limit_gb'] = result['available_ram_gb'] * 0.2
    else:
        # В консоли/exe - ограничение по RAM
        result['limit_source'] = 'ram'
        result['effective_limit_gb'] = result['available_ram_gb']

    # 4. Применяем режим (коэффициент использования)
    safety_factor = result['safety_factors'].get(mode, 0.5)
    result['usable_memory_gb'] = result['effective_limit_gb'] * safety_factor

    # 5. Если задан пользовательский лимит - используем его
    if custom_limit_gb:
        result['usable_memory_gb'] = custom_limit_gb
        result['limit_source'] = 'custom'

    # 6. Добавляем человеко-читаемое описание
    result['description'] = get_memory_description(result)

    return result


def get_jupyter_buffer_size_gb():
    """
    Gets max_buffer_size from Jupyter config in GB
    """
    config_path = os.path.expanduser('~/.jupyter/jupyter_notebook_config.py')
    default_buffer = 536870912  # 512 MB по умолчанию

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                # Ищем активную настройку
                pattern = r'^c\.NotebookApp\.max_buffer_size\s*=\s*(\d+)'
                for line in content.split('\n'):
                    if 'max_buffer_size' in line and not line.strip().startswith('#'):
                        match = re.search(r'=\s*(\d+)', line)
                        if match:
                            return int(match.group(1)) / 1024 ** 3

                # Если не нашли, ищем закомментированную (значение по умолчанию)
                pattern = r'#\s*c\.NotebookApp\.max_buffer_size\s*=\s*(\d+)'
                match = re.search(pattern, content)
                if match:
                    return int(match.group(1)) / 1024 ** 3
        except Exception:
            pass

    # Возвращаем значение по умолчанию
    return default_buffer / 1024 ** 3


def get_memory_description(memory_info):
    """
    Creates human-readable description
    """
    env_names = {
        'jupyter': 'Jupyter Notebook',
        'console': 'Console/Terminal',
        'exe': 'EXE application'
    }

    source_names = {
        'jupyter_buffer': 'Jupyter buffer',
        'ram': 'RAM',
        'ram_conservative': 'RAM (with safety margin)',
        'custom': 'user-defined limit'
    }

    desc = f"\nEnvironment: {env_names.get(memory_info['environment'], memory_info['environment'])}\n"
    desc += f"Mode: {memory_info['mode']}\n"
    desc += f"Available RAM: {memory_info['available_ram_gb']:.2f} GB\n"

    if memory_info['jupyter_buffer_gb']:
        desc += f"Jupyter buffer: {memory_info['jupyter_buffer_gb']:.2f} GB\n"

    desc += f"Limit ({source_names.get(memory_info['limit_source'], 'unknown')}): "
    desc += f"{memory_info['effective_limit_gb']:.2f} GB\n"
    desc += f"Using ({memory_info['mode']} mode): {memory_info['usable_memory_gb']:.2f} GB"

    return desc


def print_memory_report(memory_info):
    """
    Prints a formatted report
    """
    print("=" * 70)
    print("📊 MEMORY AVAILABILITY REPORT")
    print("=" * 70)
    print(memory_info['description'])
    print("=" * 70)


def check_memory(valid_points, n_wells, usable_memory_gb):
    """
    Determines the optimal method for calculating the influence matrix and optimization

    Args:
        valid_points: array of valid points
        n_wells: number of wells
        usable_memory_gb: available memory in GB for calculation

    Returns:
        dict: {'method': 'full', 'batch' or 'memmap_batch', 'batch_rows': int or None, 'batch_wells': int or None}
    """
    n_points = valid_points[:, 1].shape[0]

    # Размер итоговой матрицы в GB
    matrix_size_gb = n_points * n_wells * 4 / 1024 ** 3  # float32 = 4 bytes

    # Размер одной скважины в GB
    well_size_gb = n_points * 4 / 1024 ** 3

    # Размер массива для оптимизации в GB
    optimization_size_gb = matrix_size_gb * 2  # float32 * 2 arrays

    # Размер одной строки при оптимизации (weights + influence)
    row_size_gb = n_wells * 4 * 2 / 1024 ** 3  # two float32 arrays

    logger.info(f"Estimated memory required for matrix_r_ij: ~{matrix_size_gb:.2f} GB, \n"
                f"Estimated memory required for optimization: ~{optimization_size_gb:.2f} GB, \n"
                f"Limit RAM: ~{usable_memory_gb} GB")

    result = {
        'method': None,
        'batch_wells': None,
        'batch_rows': None,
        'matrix_size_gb': matrix_size_gb,
        'optimization_size_gb': optimization_size_gb,
        'usable_memory_gb': usable_memory_gb,
        'well_size_mb': well_size_gb * 1024,
        'n_points': n_points,
        'n_wells': n_wells
    }

    # Проверяем, хватит ли памяти для оптимизации
    if optimization_size_gb <= usable_memory_gb:
        result['method'] = 'full'
        result['message'] = f"Optimization array {optimization_size_gb:.2f} GB fits in memory"
    else:
        logger.info(f"Not enough memory available for fast computation.\n"
                    f"Batch processing will be used.")
        result['method'] = 'batch'
        batch_rows = int(usable_memory_gb * 0.9 / row_size_gb)
        result['batch_rows'] = max(100, min(batch_rows, 50000, valid_points.shape[0]))
        result['message'] = f"Optimization array {optimization_size_gb:.2f} GB does not fit in memory"
        # Проверяем, помещается ли вся матрица в память
        if matrix_size_gb > usable_memory_gb:
            result['method'] = 'memmap_batch'
            # Рассчитываем batch_wells
            batch_wells = int(usable_memory_gb / well_size_gb)
            result['batch_wells'] = max(50, min(batch_wells, 500, n_wells))
            result['message'] = (f"Matrix {matrix_size_gb:.2f} GB > {usable_memory_gb:.2f} GB, "
                                 f"using memmap with batch={result['batch_wells']}")
    return result


# Пример использования
if __name__ == "__main__":
    for mode in ['safe', 'auto', 'aggressive']:
        mem_info = get_available_memory(mode=mode)
        print_memory_report(mem_info)
