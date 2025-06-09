from scipy.optimize import bisect


def get_current_So(row, fluid_params, relative_permeability):
    """
    Calculates current oil saturation at wells
    """
    # Выделение необходимых параметров ОФП
    Swc, Sor, Fw, m1, Fo, m2 = (relative_permeability.Swc, relative_permeability.Sor, relative_permeability.Fw,
                                relative_permeability.m1, relative_permeability.Fo, relative_permeability.m2)
    # Выделение необходимых параметров флюидов
    mu_w, mu_o, Bw, Bo = fluid_params.mu_w, fluid_params.mu_o, fluid_params.Bw, fluid_params.Bo
    f_w = row['water_cut']

    if row.work_marker == "prod":
        # Текущая водонасыщенность
        Sw = get_sw(mu_w, mu_o, Bo, Bw, f_w, Fw, m1, Fo, m2, Swc, Sor)
        return 1 - Sw
    else:
        return Sor


def get_sw(mu_w, mu_o, Bo, Bw, f_w, Fw, m1, Fo, m2, Swc, Sor):
    """
    Computes water saturation (Sw) based on water cut (fw) by solving the inverse problem using the bisection method.
    """
    Sw_min = Swc  # нижняя граница интервала поиска решения
    Sw_max = 1 - Sor  # верхняя граница интервала поиска решения
    # проверка краевых значений
    if f_w <= get_f_w(mu_w, mu_o, Bo, Bw, Sw_min, Fw, m1, Fo, m2, Swc, Sor):
        Sw = Sw_min
    elif f_w >= get_f_w(mu_w, mu_o, Bo, Bw, Sw_max, Fw, m1, Fo, m2, Swc, Sor):
        Sw = Sw_max
    else:
        Sw = bisect(lambda Sw: f_w - get_f_w(mu_w, mu_o, Bo, Bw, Sw, Fw, m1, Fo, m2, Swc, Sor), Sw_min, Sw_max)
        # доп параметры: xtol=0.0001 необходимая точность решения, maxiter=1000 максимальное число итераций
    return Sw


def get_f_w(mu_w, mu_o, Bo, Bw, Sw, Fw, m1, Fo, m2, Swc, Sor):
    """
    Computes fractional flow of water (fw) as a function of water saturation (Sw) using the Buckley–Leverett
    """
    k_rw = get_k_corey(Fw, m1, Swc, Sor, Sw, type="water")  # ОФП по воде
    k_ro = get_k_corey(Fo, m2, Swc, Sor, Sw, type="oil")  # ОФП по нефти
    try:
        f_w = 100 / (1 + (k_ro * mu_w * Bw) / (k_rw * mu_o * Bo))
    except ZeroDivisionError:
        f_w = 0
    return f_w


def get_k_corey(F, m, Swc, Sor, Sw, type):
    """
    Computes relative phase permeability for oil/water as functions of water saturation (Sw) based on the Corey model.
    """
    if Sw > (1 - Sor) and type == "water":
        return 1
    elif Sw > (1 - Sor) and type == "oil":
        return 0
    elif Sw <= Swc and type == "water":
        return 0
    elif Sw <= Swc and type == "oil":
        return 1
    else:
        try:
            Sd = (Sw - Swc) / (1 - Sor - Swc)  # Приведенная водонасыщенность пласта
        except ZeroDivisionError:
            Sd = 1
        if type == "water":
            return F * (Sd ** m)
        elif type == 'oil':
            return F * ((1 - Sd) ** m)
